import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from typing import List
from uuid import uuid4
from dotenv import load_dotenv



# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
#pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embedding_dimension = 512 

# Initialize Pinecone
pc = Pinecone(api_key = pinecone_api_key)
# Connect to the index
index = pc.Index(pinecone_index_name)

def check_file_in_pinecone(source_filename: str) -> bool:
    dummy_vector = [0.0] * 512  # Match embedding dim
    try:
        response = index.query(
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