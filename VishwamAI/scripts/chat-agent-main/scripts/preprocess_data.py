import spacy
from tokenizers import BertWordPieceTokenizer
import os
import json

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face tokenizer
tokenizer = BertWordPieceTokenizer()

# Define input and output directories
input_dir = "data/raw"
output_dir = "data/processed"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def preprocess_text(text):
    # Use spaCy to tokenize and parse text
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Use Hugging Face tokenizer to further process tokens
    encoded = tokenizer.encode(" ".join(tokens))
    return encoded.tokens

def preprocess_file(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        text = item["text"]
        processed_text = preprocess_text(text)
        processed_data.append({
            "id": item["id"],
            "text": processed_text
        })

    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)

def main():
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            preprocess_file(input_file, output_file)

if __name__ == "__main__":
    main()
