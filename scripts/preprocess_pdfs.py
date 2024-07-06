import os
import sys
import fitz  # PyMuPDF

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_pdfs(input_dir, output_file):
    # List all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    # Initialize a list to store the extracted text
    extracted_texts = []

    # Process each PDF file
    for pdf_file in pdf_files:
        # Open the PDF file
        pdf_path = os.path.join(input_dir, pdf_file)
        document = fitz.open(pdf_path)

        # Extract text from each page
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            extracted_text = page.get_text()
            extracted_texts.append(extracted_text)
            # Print the extracted text for debugging purposes
            print(f"Extracted text from {pdf_file}, page {page_num}: {extracted_text}")

    # Write the extracted texts to the output file
    with open(output_file, 'w') as f:
        for text in extracted_texts:
            f.write(text + '\n')

if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), '../data/raw/pdfs')
    output_file = os.path.join(os.path.dirname(__file__), '../data/processed/pdfs_preprocessed.txt')
    preprocess_pdfs(input_dir, output_file)
