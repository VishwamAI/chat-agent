import torch
from datasets import load_dataset
import argparse
from vishwamai import config as vishwamai_config
from vishwamai.model import VishwamaiForCausalLM
from vishwamai.tokenizer import Tokenizer
from transformers.optimization import Adafactor
import os

def preprocess_data(tasks, tokenizer, max_length=1024):
    # Initialize lists to hold processed inputs and targets
    processed_inputs = []
    processed_targets = []

    # Iterate over each task
    for task in tasks:
        # Extract inputs and targets from the task
        inputs = task['inputs']
        targets = task['targets']

        # Tokenize and encode inputs and targets
        input_encodings = [tokenizer.encode(input_text, bos=True, eos=True) for input_text in inputs]
        target_encodings = [tokenizer.encode(target_text, bos=True, eos=True) for target_text in targets]

        # Dynamically pad sequences
        input_encodings = dynamic_pad_sequences(input_encodings, tokenizer.pad_id, max_length)
        target_encodings = dynamic_pad_sequences(target_encodings, tokenizer.pad_id, max_length)

        # Append processed encodings to the lists
        processed_inputs.extend(input_encodings)
        processed_targets.extend(target_encodings)

    # Convert lists to tensors
    processed_inputs = torch.tensor(processed_inputs)
    processed_targets = torch.tensor(processed_targets)

    return processed_inputs, processed_targets

def dynamic_pad_sequences(sequences, pad_id, max_length):
    # Find the longest sequence
    longest_seq = max(len(seq) for seq in sequences)
    # Pad sequences to the length of the longest sequence or max_length, whichever is smaller
    padded_length = min(longest_seq, max_length)
    return [seq + [pad_id] * (padded_length - len(seq)) for seq in sequences]

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {checkpoint_path}")
        return epoch, loss
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

def main(args):
    # Load the datasets
    datasets = [load_dataset(dataset_name) for dataset_name in args.datasets]

    # Initialize the tokenizer and model
    tokenizer_model_path = 'vishwamai/tokenizer/tokenizer.model'
    if not os.path.isfile(tokenizer_model_path):
        raise FileNotFoundError(f"Tokenizer model file not found: {tokenizer_model_path}")

    tokenizer = Tokenizer(tokenizer_model_path)
    config = vishwamai_config.get_model_config(args.model_size)
    model = VishwamaiForCausalLM(config)

    # Preprocess the data for all tasks
    tasks = [{'inputs': dataset['train']['input'], 'targets': dataset['train']['target']} for dataset in datasets]
    input_encodings, target_encodings = preprocess_data(tasks, tokenizer, args.max_length)

    # Define the optimizer and loss function
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _ = load_checkpoint(model, optimizer, args.checkpoint_path)

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
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

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch, loss.item(), args.checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Names of the datasets from Hugging Face datasets library")
    parser.add_argument("--model_size", type=str, required=True, choices=['2b', '7b', '9b', '27b'], help="Size of the Vishwamai model to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for padding")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a specific checkpoint to load")
    args = parser.parse_args()
    main(args)
