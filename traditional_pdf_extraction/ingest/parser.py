import os
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from dataclasses import dataclass
from embed.check_pinecone import check_file_in_pinecone

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, str]

class PDFIngestor:
    def __init__(self, data_dir: str, chunk_size: int = 1200, chunk_overlap: int = 300):
        self.data_dir = data_dir
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def _load_pdf(self, path: str) -> List[str]:
        """Extract raw text from each page of the PDF."""
        doc = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        return pages

    def _chunk_text(self, text: str, metadata: Dict[str, str]) -> List[DocumentChunk]:
        """Split long text into overlapping semantic chunks."""
        chunks = self.chunker.split_text(text)
        return [DocumentChunk(content=c, metadata=metadata) for c in chunks]

    def ingest(self) -> List[DocumentChunk]:
        """Main ingestion loop for all PDFs in the data directory."""
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]

        for pdf in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.data_dir, pdf)
            
            if check_file_in_pinecone(pdf):
                print(f"Skipping '{pdf}' — already in Pinecone.")
                continue

            try:
                pages = self._load_pdf(pdf_path)
                full_text = "\n\n".join(pages)
                metadata = {
                    "source": pdf,
                    "page_count": str(len(pages))
                }
                chunks = self._chunk_text(full_text, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to process {pdf}: {e}")

        return all_chunks
