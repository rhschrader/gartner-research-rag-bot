# gartner-research-rag-bot
Building a simple POC implementing RAG across a repository of Gartner research articles. My goal is two-fold: a learning experience building a RAG application, and comparison between 2 methods of context parsing.

# Solutions
## Solution 1: Parsing PDFs with LLMs
This method is inspired by thsis [OpenAI cookbook article](https://cookbook.openai.com/examples/parse_pdf_docs_for_rag). PDFs are converted to images (each page is an image) and fed to gpt-5-mini to extract all information and produce a summary. The page summary is embedded and stored in a vector database.

## Solution 2: Traditional PDF Parsing





# Technical Details
### Models
* Embedding Model: text-embedding-3-small
* Image Comprehension/Analysis: gpt-5-mini
* Chat: gpt-5-mini

### Vector Database
* Pinecone (free tier)

### Data Sources
* [SharePoint](https://wwt.sharepoint.com/:f:/r/sites/SolutionServices/Templates%20Repository/Gartner%20Repository/2025%20Gartner%20Repository?csf=1&web=1&e=eL9roQ)


