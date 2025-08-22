import os
from pinecone import Pinecone
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Constants
EMBEDDING_DIM = 512  # Must match what you used for chunk storage
TOP_K = 5            # Number of chunks to retrieve

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def get_query_embedding(query: str) -> List[float]:
    """Get OpenAI embedding for the user's query."""
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
        dimensions=EMBEDDING_DIM
    )
    return response.data[0].embedding


def retrieve_context(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Retrieve top-k matching chunks from Pinecone using embedded query."""
    query_vector = get_query_embedding(query)

    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        matches = results.get("matches", []) if isinstance(results, dict) else results.matches
        return [match.metadata for match in matches]
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []
