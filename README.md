<<<<<<< HEAD
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


=======
# Gartner Research RAG Bot

This project is a proof-of-concept for a Retrieval-Augmented Generation (RAG) system that operates on a repository of Gartner research articles. The primary goal of this project is twofold: to serve as a learning experience in building a RAG application and to compare two distinct methods of context parsing from PDF documents.

## Project Overview

The Gartner Research RAG Bot is designed to provide a conversational interface for querying a collection of Gartner research articles. It leverages the power of large language models (LLMs) to understand and respond to user queries in a natural and intuitive way. The project explores and implements two different approaches to extracting information from the source PDF documents, allowing for a comparative analysis of their effectiveness.

***

## Solutions

This project implements two distinct solutions for parsing and querying the Gartner research articles:

### Solution 1: Parsing PDFs with LLMs

This innovative approach is inspired by the [OpenAI cookbook article](https://cookbook.openai.com/examples/parse_pdf_docs_for_rag). In this method, each page of a PDF is converted into an image. These images are then fed to a multimodal LLM (`gpt-5-mini`) to extract all relevant information and generate a summary. The summary for each page is then embedded and stored in a vector database for efficient retrieval.

### Solution 2: Traditional PDF Parsing

This solution employs a more conventional method of PDF parsing. It directly extracts text from the PDF files and then splits the text into chunks. These chunks are then embedded and stored in a vector database, similar to the first solution. This approach allows for a baseline comparison against the LLM-based parsing method.

***

## Technical Details

### Models

* **Embedding Model**: `text-embedding-3-small`
* **Image Comprehension/Analysis**: `gpt-5-mini`
* **Chat**: `gpt-5-mini`

### Vector Database

* **Pinecone** (free tier)

### Data Sources

* [SharePoint](https://wwt.sharepoint.com/:f:/r/sites/SolutionServices/Templates%20Repository/Gartner%20Repository/2025%20Gartner%20Repository?csf=1&web=1&e=eL9roQ)

***

## Getting Started

To get started with this project, you will need to have Python installed on your system. You will also need to have access to the required API keys for OpenAI and Pinecone.

### Prerequisites

* Python 3.8 or higher
* An OpenAI API key
* A Pinecone API key

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/rhschrader/gartner-research-rag-bot.git](https://github.com/rhschrader/gartner-research-rag-bot.git)
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your environment variables. You will need to create a `.env` file in the root directory of the project and add the following variables:

    ```
    OPENAI_API_KEY="your_openai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    ```

***

## USAGE

This project includes two main workflows, one for each of the parsing solutions.

### Solution 1: LLM Image Processing

To run the LLM image processing workflow, you can use the following command:

```bash
python llm_image_processing/pdf_workflow.py
```
This will process the PDFs, generate the embeddings, and store them in the Pinecone vector database.

### Solution 2: Traditional PDF Extraction
To run the traditional PDF extraction workflow, you can use the following command:
```bash
python traditional_pdf_extraction/ingest_and_upsert.py
```
This will parse the PDFs, create text chunks, and store the embeddings in Pinecone.

## Chat Interface
Once the data has been processed and stored, you can interact with the RAG bot using the provided Streamlit application:
```bash
streamlit run llm_image_processing/streamlit_app_columns.py
```
or
```bash
streamlit run traditional_pdf_extraction/streamlit_app.py
```
## Future Work
Implement a more robust evaluation framework to quantitatively compare the performance of the two parsing methods.

Experiment with different embedding models and LLMs to see how they affect the overall performance of the system.

Add support for other document formats, such as DOCX and PPTX.

Develop a more sophisticated user interface with features like conversation history and document highlighting.
>>>>>>> llm_image_processing
