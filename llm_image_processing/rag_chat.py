import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import re
import pandas as pd
from prompts.rag_prompt import rag_prompt

# Load environment variables
load_dotenv()

class Rag_Chat:
    def __init__(self, top_k = 5, embedding_dimension = 1536, embedding_model = 'text-embedding-3-small'):
        self.top_k = top_k
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self.rag_prompt = rag_prompt
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
            citation = f"[Source: {source}, Page: {page}]"
            citations.append({"source":source, "page":page})
            context_lines.append(f"Citation: {citation}\nText: {text}")
        context_text = "\n\n---\n\n".join(context_lines)
        prompt = f"INPUT PROMPT:\n{query}\n-------\nCONTENT:\n{context_text}"

        return prompt, citations

    def generate_answer(self, query):
        context = self.retrieve_context(query)
        prompt, citations = self.build_prompt(context, query)


        messages = [{"role": "system", "content": self.rag_prompt}, {"role": "user", "content": prompt}]


        response = self.client.responses.create(
            model="gpt-5",
            input=messages,
            previous_response_id = self.previous_response_id
        )
        self.previous_response_id = response.id

        answer = response.output_text

        return answer, citations

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
