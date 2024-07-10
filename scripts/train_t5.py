import torch
from datasets import load_dataset
import argparse
from vishwamai import config as vishwamai_config
from vishwamai.model import VishwamaiForCausalLM
from vishwamai.tokenizer import Tokenizer
import os

def preprocess_data(data, tokenizer, max_length=512):
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    input_encodings = [tokenizer.encode(input_text, bos=True, eos=True) for input_text in inputs]
    target_encodings = [tokenizer.encode(target_text, bos=True, eos=True) for target_text in targets]

    # Pad sequences to the same length
    def pad_sequences(sequences, pad_id, max_length):
        return [seq + [pad_id] * (max_length - len(seq)) for seq in sequences]

    input_encodings = pad_sequences(input_encodings, tokenizer.pad_id, max_length)
    target_encodings = pad_sequences(target_encodings, tokenizer.pad_id, max_length)

    input_encodings = torch.tensor(input_encodings)
    target_encodings = torch.tensor(target_encodings)

    return input_encodings, target_encodings

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Initialize the tokenizer and model
    tokenizer_model_path = 'vishwamai/tokenizer/tokenizer.model'
    if not os.path.isfile(tokenizer_model_path):
        raise FileNotFoundError(f"Tokenizer model file not found: {tokenizer_model_path}")

    tokenizer = Tokenizer(tokenizer_model_path)
    config = vishwamai_config.get_model_config(args.model_size)
    model = VishwamaiForCausalLM(config)

    # Preprocess the data
    input_encodings, target_encodings = preprocess_data(dataset['train'], tokenizer)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for i in range(0, len(input_encodings), args.batch_size):
            input_ids = input_encodings[i:i+args.batch_size]
            labels = target_encodings[i:i+args.batch_size]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset from Hugging Face datasets library")
    parser.add_argument("--model_size", type=str, required=True, choices=['2b', '7b', '9b', '27b'], help="Size of the Vishwamai model to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()
    main(args)
