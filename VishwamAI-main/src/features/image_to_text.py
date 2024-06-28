import pytesseract
from PIL import Image

def image_to_text(image_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error: {e}"
