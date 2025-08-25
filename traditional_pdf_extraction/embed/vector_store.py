import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from uuid import uuid4
from ingest.parser import DocumentChunk

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
#pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embedding_dimension = 512 

# Initialize Pinecone
pc = Pinecone(api_key = pinecone_api_key)

client = OpenAI(api_key = openai_key)

# Create index if needed
if not pc.has_index(pinecone_index_name):
    print(f"Creating Pinecone index '{pinecone_index_name}' with dimension {embedding_dimension}...")
    pc.create_index(
        name=pinecone_index_name,
        dimension=embedding_dimension,
        metric = "cosine",
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the index
index = pc.Index(pinecone_index_name)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
        dimensions=512
    )
    return response.data[0].embedding

def upsert_chunks_to_pinecone(chunks: List[DocumentChunk]):
    """Convert chunks to embeddings and upsert into pc"""
    vectors = []
    for chunk in chunks:
        try:
            emb = get_embedding(chunk.content)
            vector = {
                "id": str(uuid4()),
                "values": emb,
                "metadata": {
                    "text": chunk.content,  # Optional truncation
                    **chunk.metadata
                }
            }
            vectors.append(vector)
        except Exception as e:
            print(f"Failed to embed chunk: {e}")
            continue

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted {i + len(batch)} / {len(vectors)} vectors")

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

