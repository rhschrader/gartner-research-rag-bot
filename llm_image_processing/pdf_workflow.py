from convert_pptx import convert_all_pptx_to_pdf
from llm_pdf_parser import LLM_PDF_Processor
from embed import Embed_Upsert
import os

data_dir = '/mnt/data/gartner-research-rag-bot'

# Step 0: Initialize LLM_PDF_Processor and Embed_Upsert classes
parser = LLM_PDF_Processor(data_dir)
embedder = Embed_Upsert()

# Step 1: Make sure only pdfs in directory
convert_all_pptx_to_pdf(data_dir)

# Step 2: Loop through fnames
total_files = len(os.listdir(data_dir))
counter = 1

for fname in os.listdir(data_dir):

    fpath = os.path.join(data_dir, fname)
    print(f"\n\n----------------------\nOn file {counter} / {total_files}\n----------------------\n\n")

    # Check that it is a file and is a pdf
    if os.path.isfile(fpath) & (fname.split(".")[-1] == "pdf"):
        
        # Check that the file isn't already in Pinecone
        if embedder.check_file_in_pinecone(fname):
            continue
        
        else:
            # Step 3: Create images of the pdf file
            print(f"\nCreating images of {fname} ... \n")
            images = parser.convert_doc_to_images(fpath)

            # Step 4: Save images of the pdf to return later in responses
            print(f"\nSaving {len(images)} images of {fname} ...\n")
            images_high_res = parser.convert_doc_to_images(path, dpi=200)
            paths = parser.save_pdf_images(images_high_res, fname)

            # Step 5: Analyze images of the pdf file using an LLM
            print(f"\nAnalyzing each page of {fname} and creating an AI-description ...\n")
            pages_description = parser.analyze_document(images, fname)

            # Step 6: Embed each description
            print(f"\nEmbedding each page description of {fname} ...\n")
            pages_embeddings = embedder.embed_pdf(pages_description)

            # Step 7: Upsert embeddings to pinecone
            print(f"\nUpserting to Pinecone\n")
            embedder.upsert_to_pinecone(pages_description)
    
    counter += 1









