import os
import fitz  # PyMuPDF
from pathlib import Path

def render_pdf_page(pdf_name: str, page_number: int, data_dir: str = "/domino/datasets/local/Gartner_Article_Chat", output_dir: str = "page_images") -> str:
    """
    Render a specific page from a PDF (by filename) to an image.
    
    Args:
        pdf_name: PDF file name (e.g. "report.pdf")
        page_number: 1-based page number to render
        data_dir: Directory where PDFs are stored
        output_dir: Directory where images will be saved

    Returns:
        Path to the saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(data_dir, pdf_name)
    doc = fitz.open(pdf_path)

    page_index = page_number - 1
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Invalid page number: {page_number} for {pdf_name}")

    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(1,1))  # 2x zoom for better resolution

    image_filename = f"{Path(pdf_name).stem}_page_{page_number}.png"
    image_path = os.path.join(output_dir, image_filename)
    pix.save(image_path)

    return image_path
