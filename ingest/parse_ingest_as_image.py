# Imports
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
import io
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval
from dataclasses import dataclass
from embed.check_pinecone import check_file_in_pinecone

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, str]

class PDFIngestor:
    def __init__(self, data_dir: str, chunk_size: int = 1200, chunk_overlap: int = 300):
        self.data_dir = data_dir
    
    def 

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



def convert_doc_to_images(path, chunk_size=2):
    
    reader = PdfReader(path)
    total_pages = len(reader.pages)

    all_images = []
    for start_page in range(1, total_pages + 1, chunk_size):
        end_page = min(start_page + chunk_size - 1, total_pages)
        images = convert_from_path(
            path,
            first_page=start_page,
            last_page=end_page,
            dpi = 50
        )
        all_images.extend(images)

    return all_images
    return images

def extract_text_from_doc(path):
    text = extract_text(path)
    return text