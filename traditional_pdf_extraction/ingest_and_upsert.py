from ingest.convert_pptx import convert_all_pptx_to_pdf
from ingest.parser import PDFIngestor
from embed.vector_store import upsert_chunks_to_pinecone

data_path = '/mnt/data/gartner-research-rag-bot'

convert_all_pptx_to_pdf(data_path)         # Step 1: Normalize format
ingestor = PDFIngestor(data_path)     # Step 2: Parse & chunk PDFs
chunks = ingestor.ingest()
upsert_chunks_to_pinecone(chunks)          # Step 3: Embed + upsert
