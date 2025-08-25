import os
import subprocess

def convert_all_pptx_to_pdf(data_dir: str):
    for fname in os.listdir(data_dir):
        # check if it is a powerpoint or word file
        if fname.endswith(".pptx") or fname.endswith(".docx"):
            pptx_docx_path = os.path.join(data_dir, fname)
            if fname.endswith(".pptx"):
                pdf_name = fname.replace(".pptx", ".pdf")
            else:
                pdf_name = fname.replace(".docx", ".pdf")
            pdf_path = os.path.join(data_dir, pdf_name)

            if os.path.exists(pdf_path):
                # Delete the pptx file if a PDF with the same name already exists
                print(f"📄 PDF already exists for {fname}. Deleting the PPTX or DOCX file.")
                os.remove(pptx_docx_path)

            else:
                try:
                    subprocess.run(
                        ["libreoffice", "--headless", "--convert-to", "pdf", pptx_docx_path, "--outdir", data_dir],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"✅ Converted: {fname} → {pdf_name}")
                    
                    # Delete the powerpoint or word file after successful conversion
                    os.remove(pptx_docx_path)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to convert {fname}: {e}")
