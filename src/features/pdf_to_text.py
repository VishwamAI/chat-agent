import fitz  # PyMuPDF

def pdf_to_text(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error: {e}"
