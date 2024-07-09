import torch
from transformers import T5Tokenizer
from datasets import load_dataset
import argparse
from vishwamai import config as vishwamai_config
from vishwamai.model import VishwamaiForCausalLM

def preprocess_data(data, tokenizer, max_length=512):
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    input_encodings = tokenizer(inputs, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    target_encodings = tokenizer(targets, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input_encodings, target_encodings

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    config = vishwamai_config.get_model_config('7b')
    model = VishwamaiForCausalLM(config)

    # Preprocess the data
    input_encodings, target_encodings = preprocess_data(dataset['train'], tokenizer)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for i in range(0, len(input_encodings.input_ids), args.batch_size):
            input_ids = input_encodings.input_ids[i:i+args.batch_size]
            attention_mask = input_encodings.attention_mask[i:i+args.batch_size]
            labels = target_encodings.input_ids[i:i+args.batch_size]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset from Hugging Face datasets library")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()
    main(args)
