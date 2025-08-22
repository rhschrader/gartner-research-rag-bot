import os
import subprocess

def convert_all_pptx_to_pdf(data_dir: str):
    for fname in os.listdir(data_dir):
        if fname.endswith(".pptx"):
            pptx_path = os.path.join(data_dir, fname)
            pdf_name = fname.replace(".pptx", ".pdf")
            pdf_path = os.path.join(data_dir, pdf_name)

            if os.path.exists(pdf_path):
                # Delete the pptx file if a PDF with the same name already exists
                print(f"📄 PDF already exists for {fname}. Deleting the PPTX file.")
                os.remove(pptx_path)

            else:
                try:
                    subprocess.run(
                        ["libreoffice", "--headless", "--convert-to", "pdf", pptx_path, "--outdir", data_dir],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"✅ Converted: {fname} → {pdf_name}")
                    # Delete the pptx file after successful conversion
                    os.remove(pptx_path)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to convert {fname}: {e}")
