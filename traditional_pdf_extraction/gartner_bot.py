import os
from ingest.convert_pptx import convert_all_pptx_to_pdf
from ingest.parser import PDFIngestor
from embed.vector_store import upsert_chunks_to_pinecone
from query.rag_chat import generate_answer

DATA_DIR = '/domino/datasets/local/Gartner_Article_Chat'

def run_ingestion():
    print("📂 Converting PowerPoints to PDFs (if needed)...")
    convert_all_pptx_to_pdf(DATA_DIR)

    print("📚 Parsing and chunking documents...")
    ingestor = PDFIngestor(DATA_DIR)
    chunks = ingestor.ingest()

    if chunks:
        print(f"🧠 Found {len(chunks)} new chunks. Embedding and upserting...")
        upsert_chunks_to_pinecone(chunks)
    else:
        print("✅ No new documents to process. Skipping ingestion.")

def run_chat_loop():
    print("\n💬 GartnerBot is ready! Ask a question, or type 'exit' to quit.\n")
    while True:
        query = input("🔍 Ask: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        print("⏳ Thinking...\n")
        answer = generate_answer(query)
        print("🧠 Answer:\n", answer, "\n")

def app_chat_response(prompt: str, history: list) -> tuple[str, list]:
    return generate_answer(prompt, history)

if __name__ == "__main__":
    run_ingestion()
    run_chat_loop()
