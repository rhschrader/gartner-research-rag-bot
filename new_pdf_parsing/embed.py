import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm
#import concurrent.futures

# Load environment variables
load_dotenv()


class Embedder:
    def __init__(self, embedding_dimension = 1536, embedding_model = 'text-embedding-3-small'):
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self.initialize_pinecone()
        self.initialize_openai()
        
    def initialize_pinecone(self):
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_TRADITIONAL_INDEX_NAME")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key = pinecone_api_key)
        # Connect to the index
        self.index = self.pc.Index(pinecone_index_name)
    
    def initialize_openai(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key = openai_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-small",
                dimensions=self.embedding_dimension
            )
            vecs.append(response.data[0].embedding)
        return vecs
    

    def check_file_in_pinecone(self, source_filename: str) -> bool:
        dummy_vector = [0.0] * self.embedding_dimension  # Match embedding dim
        try:
            response = self.index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=True,
                filter={"source": source_filename}
            )
            matches = response.get("matches", [])
            return len(matches) > 0
        except Exception as e:
            print(f"Failed to check for existing file: {e}")
            return False

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small",
            dimensions=self.embedding_dimension
        )
        return response.data[0].embedding

    # Come back to make this concurrent
    def embed_pdf(self, pages_description):
        for page in tqdm(pages_description):
            page['embedding'] = self.get_embedding(page['description'])

        return pages_description

    def upsert(self, embs, payloads):
        vectors = []
        for i in range(len(embs)):
            vector = {
                "id": str(uuid4()),
                "values": embs[i],
                "metadata": payloads[i]
            }
            vectors.append(vector)
        vector
        # Upsert PDF data
        self.index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors")        