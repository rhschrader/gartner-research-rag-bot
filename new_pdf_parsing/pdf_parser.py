"""
Enterprise-Ready PDF → Chunks pipeline for RAG

Key features
- Robust extraction via PyMuPDF (primary) with OCR fallback (Tesseract)
- Optional tables extraction via pdfplumber
- Layout-aware heading detection (font-size heuristics)
- Header/Footer removal (repeated line detection)
- Semantic-ish hierarchical chunking with overlap
- Rich metadata per chunk (doc_id, page, section, coordinates, version)
- Pluggable vector store interface + simple local example
- CLI for batch processing of a folder → JSONL for downstream indexing

Dependencies
    pip install pymupdf pdfplumber pillow pytesseract numpy rapidfuzz tiktoken

System deps
    Tesseract OCR must be installed and on PATH (https://tesseract-ocr.github.io/)

Notes
- Tables are parsed best-effort; complex tables may require domain-specific parsers.
- Token counting uses tiktoken if available, else a heuristic fallback.
"""
from __future__ import annotations

import os
import io
import re
import json
import uuid
import time
import math
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Optional, Tuple
from embed import Embedder

import numpy as np

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required: pip install pymupdf") from e

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESS = True
except Exception:
    HAS_TESS = False

try:
    import tiktoken
    def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text or ""))
except Exception:
    def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        # Heuristic: ~4 chars/token (English) → conservative fallback
        if not text:
            return 0
        return max(1, int(len(text) / 4))

# --------------------------- Models --------------------------- #

@dataclass
class Span:
    text: str
    bbox: Tuple[float, float, float, float]
    size: float
    bold: bool
    italic: bool

@dataclass
class Block:
    text: str
    spans: List[Span]
    bbox: Tuple[float, float, float, float]
    kind: str  # 'heading' | 'paragraph' | 'list' | 'table' | 'ocr' | 'figure' | 'unknown'

@dataclass
class PageExtraction:
    page_number: int  # 1-indexed
    blocks: List[Block]
    width: float
    height: float

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    section_path: List[str]
    text: str
    metadata: Dict[str, Any]

# --------------------------- Utilities --------------------------- #

def stable_doc_id(path: str) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(path).encode("utf-8"))
    try:
        h.update(str(os.path.getmtime(path)).encode("utf-8"))
    except Exception:
        pass
    return h.hexdigest()[:16]

HEADER_FOOTER_MAX_LINES = 3

class HeaderFooterCleaner:
    """Detects repeated header/footer lines across pages and removes them."""
    def __init__(self):
        self.top_counts: Dict[str, int] = {}
        self.bot_counts: Dict[str, int] = {}

    @staticmethod
    def _norm(line: str) -> str:
        s = re.sub(r"\s+", " ", (line or "").strip())
        return s.lower()

    def scan_page(self, page_text: str):
        lines = [l for l in page_text.splitlines() if l.strip()]
        if not lines:
            return
        top = lines[:HEADER_FOOTER_MAX_LINES]
        bot = lines[-HEADER_FOOTER_MAX_LINES:]
        for t in top:
            k = self._norm(t)
            self.top_counts[k] = self.top_counts.get(k, 0) + 1
        for b in bot:
            k = self._norm(b)
            self.bot_counts[k] = self.bot_counts.get(k, 0) + 1

    def clean(self, page_text: str, min_pages: int) -> str:
        lines = page_text.splitlines()
        out = []
        for i, l in enumerate(lines):
            nl = self._norm(l)
            is_top = i < HEADER_FOOTER_MAX_LINES and self.top_counts.get(nl, 0) >= min_pages // 2
            is_bot = i >= len(lines) - HEADER_FOOTER_MAX_LINES and self.bot_counts.get(nl, 0) >= min_pages // 2
            if is_top or is_bot:
                continue
            out.append(l)
        return "\n".join(out)

# --------------------------- Extraction --------------------------- #

class PDFExtractor:
    def __init__(self, ocr_threshold_chars: int = 40, ocr_lang: str = "eng"):
        self.ocr_threshold_chars = ocr_threshold_chars
        self.ocr_lang = ocr_lang
        if not HAS_TESS:
            logging.warning("pytesseract not found; OCR fallback disabled.")

    def _extract_page_pymupdf(self, page: fitz.Page) -> PageExtraction:
        raw = page.get_text("dict")
        blocks: List[Block] = []
        for b in raw.get("blocks", []):
            if "lines" not in b:
                # could be an image block etc.
                if b.get("type", 0) == 1:
                    kind = "figure"
                else:
                    kind = "unknown"
                blocks.append(Block(text="", spans=[], bbox=tuple(b.get("bbox", (0,0,0,0))), kind=kind))
                continue
            texts = []
            spans: List[Span] = []
            for line in b.get("lines", []):
                for sp in line.get("spans", []):
                    t = sp.get("text", "")
                    if t:
                        texts.append(t)
                    spans.append(Span(
                        text=t,
                        bbox=tuple(sp.get("bbox", (0,0,0,0))),
                        size=float(sp.get("size", 0)),
                        bold="bold" in (sp.get("font", "").lower()),
                        italic="italic" in (sp.get("font", "").lower())
                    ))
            text = " ".join(texts)
            blocks.append(Block(text=text, spans=spans, bbox=tuple(b.get("bbox", (0,0,0,0))), kind="paragraph"))
        return PageExtraction(page.number + 1, blocks, page.rect.width, page.rect.height)

    def _ocr_page(self, page: fitz.Page) -> Optional[Block]:
        if not HAS_TESS:
            return None
        # Render at 2x for better OCR quality
        zoom = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=zoom, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
        if text and text.strip():
            return Block(text=text, spans=[], bbox=(0,0,pix.width,pix.height), kind="ocr")
        return None

    def extract(self, pdf_path: str) -> List[PageExtraction]:
        doc = fitz.open(pdf_path)
        pages: List[PageExtraction] = []
        cleaner = HeaderFooterCleaner()

        # First pass: extract and detect headers/footers
        tmp_texts = []
        for p in doc:
            pe = self._extract_page_pymupdf(p)
            full_text = "\n".join([b.text for b in pe.blocks if b.text])
            tmp_texts.append(full_text)
            cleaner.scan_page(full_text)
            pages.append(pe)

        # OCR pass for text-light pages
        for i, p in enumerate(doc):
            page_text_len = len(tmp_texts[i] or "")
            if page_text_len < self.ocr_threshold_chars:
                ocr_block = self._ocr_page(p)
                if ocr_block:
                    pages[i].blocks.append(ocr_block)
                    tmp_texts[i] = (tmp_texts[i] + "\n" + ocr_block.text).strip()

        # Clean headers/footers and reassign block texts
        min_pages = max(2, len(doc))
        for i, pe in enumerate(pages):
            joined = "\n".join([b.text for b in pe.blocks if b.text])
            cleaned = cleaner.clean(joined, min_pages)
            # Distribute cleaned text back by simple proportion
            if not cleaned:
                continue
            # naive: replace paragraph blocks' text from cleaned lines (best-effort)
            lines = cleaned.splitlines()
            j = 0
            for b in pe.blocks:
                if b.kind in {"paragraph", "ocr"}:
                    take = min(len(lines) - j, 10)  # arbitrary segment
                    b.text = "\n".join(lines[j:j+take])
                    j += take

        doc.close()
        return pages

# --------------------------- Heading Detection --------------------------- #

class HeadingDetector:
    def detect(self, pe: PageExtraction) -> List[Block]:
        # Determine heading threshold per page
        sizes = []
        for b in pe.blocks:
            for s in b.spans:
                sizes.append(s.size)
        thresh = np.percentile(sizes, 85) if sizes else 0

        out: List[Block] = []
        for b in pe.blocks:
            if not b.spans:
                out.append(b)
                continue
            avg_size = np.mean([s.size for s in b.spans])
            is_heading = avg_size >= thresh and len(b.text.strip()) <= 200
            kind = "heading" if is_heading else b.kind
            out.append(Block(text=b.text, spans=b.spans, bbox=b.bbox, kind=kind))
        return out

# --------------------------- Table Extraction --------------------------- #

class TableExtractor:
    def extract_tables(self, pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
        tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
        if not HAS_PDFPLUMBER:
            return tables_by_page
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                    tjson = []
                    for t in tables:
                        tjson.append({"rows": t})
                    if tjson:
                        tables_by_page[i] = tjson
                except Exception:
                    continue
        return tables_by_page

# --------------------------- Chunking --------------------------- #

class HierarchicalChunker:
    def __init__(self, max_tokens: int = 800, overlap_tokens: int = 120):
        assert overlap_tokens < max_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter
        s = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[\"])", text.strip())
        return [x.strip() for x in s if x.strip()]

    def _greedy_pack(self, sentences: List[str]) -> List[str]:
        chunks: List[str] = []
        cur: List[str] = []
        cur_tokens = 0
        for sent in sentences:
            stoks = count_tokens(sent)
            if cur and cur_tokens + stoks > self.max_tokens:
                chunks.append(" ".join(cur).strip())
                # overlap
                if self.overlap_tokens > 0 and chunks[-1]:
                    back = []
                    btoks = 0
                    for s in reversed(cur):
                        t = count_tokens(s)
                        if btoks + t > self.overlap_tokens:
                            break
                        back.append(s)
                        btoks += t
                    cur = list(reversed(back))
                    cur_tokens = sum(count_tokens(s) for s in cur)
                else:
                    cur = []
                    cur_tokens = 0
            cur.append(sent)
            cur_tokens += stoks
        if cur:
            chunks.append(" ".join(cur).strip())
        return chunks

    def chunk(self, pages: List[PageExtraction], doc_id: str, tables: Dict[int, List[Dict[str, Any]]] | None = None) -> List[Chunk]:
        tables = tables or {}
        chunks: List[Chunk] = []
        section_stack: List[str] = []
        buff: List[Tuple[int, str, List[str]]] = []  # (page_no, text, section_path)

        # Build a joined stream with headings guiding sections
        hd = HeadingDetector()
        for pe in pages:
            blocks = hd.detect(pe)
            for b in blocks:
                if b.kind == "heading" and b.text.strip():
                    # start new section
                    section_stack.append(b.text.strip())
                elif b.kind in {"paragraph", "ocr"} and b.text.strip():
                    buff.append((pe.page_number, b.text.strip(), list(section_stack)))
            # Attach tables as dedicated chunks
            if pe.page_number in tables:
                for t in tables[pe.page_number]:
                    ttext = self._table_to_markdown(t.get("rows", []))
                    if ttext.strip():
                        buff.append((pe.page_number, ttext, list(section_stack) + ["Table"]))

        # Now pack into token-limited chunks respecting sentence boundaries
        i = 0
        while i < len(buff):
            page_no, text, sec = buff[i]
            sentences = self._split_sentences(text)
            packed = self._greedy_pack(sentences)
            for p in packed:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page_start=page_no,
                    page_end=page_no,
                    section_path=sec,
                    text=p,
                    metadata={
                        "page": page_no,
                        "section_path": sec,
                        "token_count": count_tokens(p),
                    }
                ))
            i += 1

        return chunks

    @staticmethod
    def _table_to_markdown(rows: List[List[str]]) -> str:
        if not rows:
            return ""
        # Assume first row is header if all cells are non-empty
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        md = []
        md.append(" | ".join(str(c or "").strip() for c in header))
        md.append(" | ".join(["---"] * len(header)))
        for r in body:
            md.append(" | ".join(str(c or "").strip() for c in r))
        return "\n".join(md)

# --------------------------- Vector Store Interface --------------------------- #
"""
class VectorStore:
    def upsert(self, embeddings: List[List[float]], payloads: List[Dict[str, Any]]):
        raise NotImplementedError
"""
# --------------------------- Embeddings Adapter --------------------------- #

## from embed.py

# --------------------------- Pipeline --------------------------- #

class RAGIngestionPipeline:
    def __init__(self, embedder: Embedder, max_tokens: int = 800, overlap_tokens: int = 120):
        self.extractor = PDFExtractor()
        self.tables = TableExtractor()
        self.chunker = HierarchicalChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.embedder = embedder
        #self.vstore = vstore

    def process_pdf(self, path: str) -> List[Chunk]:
        doc_id = stable_doc_id(path)
        pages = self.extractor.extract(path)
        tmap = self.tables.extract_tables(path)
        chunks = self.chunker.chunk(pages, doc_id=doc_id, tables=tmap)
        doc_name = os.path.basename(path)
        # Attach common metadata
        for c in chunks:
            c.metadata.update({
                "doc_name":doc_name
            })
        return chunks

    def index_chunks(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        embs = self.embedder.embed(texts)
        payloads = [
            {
                "text": c.text,
                **c.metadata,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in chunks
        ]
        self.embedder.upsert(embs, payloads)

    def export_jsonl(self, chunks: List[Chunk], out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                rec = {
                    "id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_path": c.section_path,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------- CLI --------------------------- #

def _iter_pdfs(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                yield os.path.join(dirpath, fn)

def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="PDF → RAG chunks pipeline")
    parser.add_argument("input", default='/mnt/data/gartner-research-rag-bot', help="PDF file or directory")
    parser.add_argument("--out", default="chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--overlap_tokens", type=int, default=120)
    parser.add_argument("--index", action="store_true", help="Also index into a local demo store")
    args = parser.parse_args()

    embedder = Embedder()  # Replace with your enterprise embedder
    #vstore = LocalAnnoyStore()
    pipe = RAGIngestionPipeline(embedder, args.max_tokens, args.overlap_tokens)

    paths = [args.input]
    if os.path.isdir(args.input):
        paths = list(_iter_pdfs(args.input))
    if not paths:
        print("No PDFs found.")
        return

    #all_chunks: List[Chunk] = []
    for p in paths[:5]:
        print(f"Processing {p} ...")
        chunks = pipe.process_pdf(p)
        pipe.index_chunks(chunks)
        #all_chunks.extend(chunks)

    #pipe.export_jsonl(all_chunks, args.out)
    #print(f"Exported {len(all_chunks)} chunks → {args.out}")

    #if args.index:
        #pipe.index_chunks(all_chunks)
        #print(f"Indexed {len(all_chunks)} chunks into pinecone store.")

if __name__ == "__main__":
    run_cli()
