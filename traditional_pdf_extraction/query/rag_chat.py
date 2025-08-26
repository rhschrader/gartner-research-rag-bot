import os
from openai import OpenAI
from dotenv import load_dotenv
from query.retriever import retrieve_context
from pinecone import Pinecone
import re
import pandas as pd

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

def build_prompt(context_chunks, query):
    context_lines = []
    citations = []
    for chunk in context_chunks:
        source = chunk.get("source", "Unknown document")
        page = chunk.get("page_count", "Unknown page")
        text = chunk["text"]
        citation = f"[Source: {source}, Page: {page}]"
        citations.append({"source":source, "page":page, "citation":citation})
        context_lines.append(f"Citation: {citation}\nText: {text}")
    context_text = "\n\n---\n\n".join(context_lines)
    prompt = f"""
You are an expert assistant answering questions using Gartner research. Be very professional and academic. Your answers should be reliable and trustworthy.

Use the context below to answer the user's question. You may add your own knowledge **only if** it supports the context. Be accurate and concise. When referencing the context, cite the article title and page number in-line with the context.

Context:
{context_text}

Question:
{query}

Answer:"""
    return prompt.strip(), citations


def generate_answer(query: str, history: list) -> tuple[str, list]:
    context_chunks = retrieve_context(query)
    context_text, citations = build_prompt(context_chunks, query)

    max_turns = 10
    if len(history) > max_turns * 2 + 1:
        history = [history[0]] + history[-(max_turns * 2):]
    
    messages = history + [{"role": "user", "content": context_text}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.1
    )

    answer = response.choices[0].message.content
    
    used_citations = [citation for citation in citations if citation['source'].split(' -')[0] in answer]
    print(citations)

    messages.append({"role": "assistant", "content": answer})

    return answer, messages, used_citations

