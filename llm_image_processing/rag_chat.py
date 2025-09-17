import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import re
import pandas as pd
from prompts.rag_prompt import rag_prompt
from prompts.rag_check_prompt import rag_check_prompt, rag_not_needed_prompt
import fitz  # PyMuPDF
from pathlib import Path

# Load environment variables
load_dotenv()

class Rag_Chat:
    def __init__(self, top_k = 8, embedding_dimension = 1536, embedding_model = 'text-embedding-3-small', chat_model='gpt-5-mini'):
        self.top_k = top_k
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.rag_prompt = rag_prompt
        self.rag_check_prompt = rag_check_prompt
        self.rag_not_needed_prompt = rag_not_needed_prompt
        self.initialize_pinecone()
        self.initialize_openai()
        self.history = []
        self.previous_response_id = None
    
    def initialize_pinecone(self):
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_LLM_INDEX_NAME")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key = pinecone_api_key)
        # Connect to the index
        self.index = self.pc.Index(pinecone_index_name)
    
    def initialize_openai(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key = openai_key)

    def build_prompt(self, context, query):
        context_lines = []
        citations = []
        for c in context:
            source = c['source']
            page = c['page']
            text = c['text']
            citation = f"[Source: {source}, Page: {int(page)}]"
            citations.append({"source":source, "page":page})
            context_lines.append(f"Citation: {citation}\nText: {text}")
        context_text = "\n\n---\n\n".join(context_lines)
        prompt = f"INPUT PROMPT:\n{query}\n-------\nCONTENT:\n{context_text}"

        return prompt, citations

    def generate_messages(self, query, history):
        
        check = self.rag_needed_check(query, history)

        if check == True:
            context = self.retrieve_context(query)
            prompt, citations = self.build_prompt(context, query)

            messages = [{"role": "system", "content": self.rag_prompt}, {"role": "user", "content": prompt}]
            
            return messages, citations
        else:
            messages = [{"role":"system", "content":self.rag_not_needed_prompt}, {"role": "user", "content": query}]
            return messages, None

    def generate_answer(self, query):
        context = self.retrieve_context(query)
        prompt, citations = self.build_prompt(context, query)


        messages = [{"role": "system", "content": self.rag_prompt}, {"role": "user", "content": prompt}]


        response = self.client.responses.create(
            model=self.chat_model,
            input=messages,
            previous_response_id = self.previous_response_id
        )
        self.previous_response_id = response.id

        answer = response.output_text

        return answer, citations

    def rag_needed_check(self, query, history):
        chat_input = history + [{"role": "system", "content": self.rag_check_prompt}, {"role": "user", "content": query}]
        response = self.client.responses.create(
            model=self.chat_model,
            input=chat_input
        )
        print(f"\n\nQUERY: {query}\n\nYES-NO: {response.output_text}\n\n")
        if response.output_text.lower() == "yes":
            return True
        elif response.output_text.lower() == "no":
            return False
        else:
            print(f"Error: Did not receive YES or NO\n\nReceived: {response.output_text}\n\nDefaulting to RAG\n")
            return True

    def get_query_embedding(self, query):
        response = self.client.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
            dimensions=self.embedding_dimension
        )
        return response.data[0].embedding


    def retrieve_context(self, query):
        """Retrieve top-k matching chunks from Pinecone using embedded query."""
        query_vector = self.get_query_embedding(query)

        try:
            results = self.index.query(
                vector=query_vector,
                top_k=self.top_k,
                include_metadata=True
            )
            matches = results.get("matches", []) if isinstance(results, dict) else results.matches
            return [match.metadata for match in matches]
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

    def get_pdf_image(self, citation, image_dir='/mnt/data/gartner_pdf_images'):
        source = citation['source']
        page = int(citation['page'])

        base_name = source.replace('.pdf', '')
        image_filename = f"{base_name}_page_{page}.png"
        # Construct the full path to the image
        image_path = os.path.join(image_dir, image_filename)
        print(image_path)
        
        # Check if the image exists
        if os.path.exists(image_path):
            return image_path
        else:
            print("Image path does not exist")
            return None

    def render_pdf_page(self, pdf_name: str, page_number: int, data_dir: str = "/mnt/data/gartner-research-rag-bot", output_dir: str = "page_images") -> str:
        """
        Render a specific page from a PDF (by filename) to an image.
        
        Args:
            pdf_name: PDF file name (e.g. "report.pdf")
            page_number: 1-based page number to render
            data_dir: Directory where PDFs are stored
            output_dir: Directory where images will be saved

        Returns:
            Path to the saved image
        """
        os.makedirs(output_dir, exist_ok=True)

        pdf_path = os.path.join(data_dir, pdf_name)
        doc = fitz.open(pdf_path)

        page_index = page_number - 1
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Invalid page number: {page_number} for {pdf_name}")

        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2,2))  # 2x zoom for better resolution

        image_filename = f"{Path(pdf_name).stem}_page_{page_number}.png"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)

        return image_path

    import os

    def render_pdf_page_debug(self, pdf_name, page_number, data_dir: str = "/mnt/data/gartner-research-rag-bot", output_dir: str = "page_images") -> str:
        
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(data_dir, pdf_name)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        page_index = page_number - 1
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Invalid page number: {page_number} for {pdf_name}")

        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        image_filename = f"{Path(pdf_name).stem}_page_{page_number}.png"
        image_path = os.path.join(output_dir, image_filename)

        pix.save(image_path)

        return image_path


