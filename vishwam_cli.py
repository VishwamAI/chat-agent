import argparse
import torch
from transformers import T5Tokenizer
import requests
from bs4 import BeautifulSoup
from models import Transformer, TransformerConfig

def generate_response(input_text):
    # Load the trained model and tokenizer
    config = TransformerConfig(
        vocab_size=32128,
        output_vocab_size=32128,
        emb_dim=512,
        num_heads=8,
        num_layers=6,
        qkv_dim=512,
        mlp_dim=2048,
        max_len=2048,
        dropout_rate=0.3,
        attention_dropout_rate=0.3
    )
    model = Transformer(config=config)
    model.load_state_dict(torch.load('./vishwam_model/pytorch_model.bin'))
    tokenizer = T5Tokenizer.from_pretrained('./vishwam_model')

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response using the model
    output_ids = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

    # Decode the generated response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

def fetch_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def main():
    parser = argparse.ArgumentParser(description="Vishwam Model CLI")
    parser.add_argument('--input', type=str, help='Input text for the model')
    parser.add_argument('--url', type=str, help='URL to fetch content from')
    args = parser.parse_args()

    if args.input:
        response = generate_response(args.input)
        print("Input:", args.input)
        print("Response:", response)
    elif args.url:
        content = fetch_web_content(args.url)
        print("Fetched Content:", content)
    else:
        print("Please provide either --input or --url argument")

if __name__ == "__main__":
    main()
