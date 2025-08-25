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
from pathlib import Path
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
from pdf_analyze_prompt import pdf_analyze_prompt

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, str]

class LLM_PDF_Processor:
    def __init__(self, data_dir, llm_model = 'gpt-5-mini', embedding_model = 'text-embedding-3-small', dpi = 50):
        self.data_dir = data_dir
        self.pdf_analyze_prompt = pdf_analyze_prompt
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.dpi = dpi

    def convert_doc_to_images_chunked(self, path, chunk_size=2):
    
        reader = PdfReader(path)
        total_pages = len(reader.pages)

        all_images = []
        for start_page in range(1, total_pages + 1, chunk_size):
            end_page = min(start_page + chunk_size - 1, total_pages)
            images = convert_from_path(
                path,
                first_page=start_page,
                last_page=end_page,
                dpi = self.dpi
            )
            all_images.extend(images)

        return all_images

    def convert_doc_to_images(self, path):
        images = convert_from_path(path, dpi = self.dpi)
        return images


    def extract_text_from_doc(self, path):
        text = extract_text(path)
        return text


    def analyze_image(self, data_uri):
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.pdf_analyze_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{data_uri}"
                        }
                        }
                    ]
                    },
            ]
        )
        return response.choices[0].message.content

    # Converting images to base64 encoded images in a data URI format to use with the ChatCompletions API
    def get_img_uri(self, img):
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')

        data_uri = f"data:image/png;base64,{base64_png}"
        return data_uri

    def analyze_doc_image(self, img):
        img_uri = self.get_img_uri(img)
        data = self.analyze_image(img_uri)
        return data

    # Analyze a list of pdf page images with an LLM.
    ## Return description, source, and page
    def analyze_document(self, doc_images, fname):
        """
         This file will take images of a pdf, and analyze them with GPT-5-mini. 
         Page descriptions will be embedded for RAG
        
        """
        # list to store dicts of source, page, description
        pages_description = []

        # concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            
            futures = [
                executor.submit(self.analyze_doc_image, image)
                for image in doc_images
            ]
            
            print(f"\nAnalyzing {len(doc_images)} images in {fname}\n")
            
            with tqdm(total=len(doc_images)) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            for index, f in enumerate(futures):
                res = f.result()
                pages_description.append({'source': fname, 'page': index+1, 'description': res})

        return pages_description

        ## Save images of documents
        def save_pdf_images(self, images, fname, output_dir='/mnt/data/gartner_pdf_images'):
            """
            Save a list of images with filenames that include the PDF name and page number.
            
            Args:
                image_list: List of tuples containing (image_data, page_number)
                pdf_name: Original PDF file name (e.g. "report.pdf")
                output_dir: Directory where images will be saved

            Returns:
                List of paths to the saved images
            """
            os.makedirs(output_dir, exist_ok=True)

            image_paths = []

            for index, image_data in enumerate(image_list):
                image_filename = f"{Path(pdf_name).stem}_page_{index + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                
                # Assuming image_data is a PIL Image object or similar
                image_data.save(image_path)

                image_paths.append(image_path)

            return image_paths
                