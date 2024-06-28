import os
import sys
import pytesseract
from PIL import Image

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_images(input_dir, output_file):
    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Initialize a list to store the extracted text
    extracted_texts = []

    # Process each image file
    for image_file in image_files:
        # Open the image file
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)

        # Print the name of the file being processed
        print(f"Processing file: {image_file}")

        # Use Tesseract OCR to extract text from the image
        extracted_text = pytesseract.image_to_string(image)
        extracted_texts.append(extracted_text)

        # Print the extracted text
        print(f"Extracted text: {extracted_text}")

    # Write the extracted texts to the output file
    with open(output_file, 'w') as f:
        for text in extracted_texts:
            f.write(text + '\n')

if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), '../data/raw/images')
    output_file = os.path.join(os.path.dirname(__file__), '../data/processed/images_preprocessed.txt')
    preprocess_images(input_dir, output_file)
