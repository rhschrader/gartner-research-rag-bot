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

def extract_text_from_doc(path):
    text = extract_text(path)
    return text