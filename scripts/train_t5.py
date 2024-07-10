import torch
from datasets import load_dataset
import argparse
from vishwamai import config as vishwamai_config
from vishwamai.model import VishwamaiForCausalLM
from vishwamai.tokenizer import Tokenizer
from transformers.optimization import Adafactor
import os

def preprocess_data(tasks, tokenizer, max_length=1024, device='cpu'):
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

    # Convert lists to tensors and move to the specified device
    processed_inputs = torch.tensor(processed_inputs).to(device)
    processed_targets = torch.tensor(processed_targets).to(device)

    return processed_inputs, processed_targets

def dynamic_pad_sequences(sequences, pad_id, max_length):
    # Find the longest sequence
    longest_seq = max(len(seq) for seq in sequences)
    # Pad sequences to the length of the longest sequence or max_length, whichever is smaller
    padded_length = min(longest_seq, max_length)
    return [seq + [pad_id] * (padded_length - len(seq)) for seq in sequences]

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'device': str(next(model.parameters()).device)  # Save the device information
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Move optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Checkpoint loaded from {checkpoint_path}")
            return epoch, loss
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float('inf')  # Return default values in case of error
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

def preprocess_validation_data(validation_dataset, tokenizer, max_length=1024, device='cpu'):
    # Extract inputs and targets from the validation dataset
    inputs = validation_dataset['validation']['input']
    targets = validation_dataset['validation']['target']

    # Tokenize and encode inputs and targets
    input_encodings = [tokenizer.encode(input_text, bos=True, eos=True) for input_text in inputs]
    target_encodings = [tokenizer.encode(target_text, bos=True, eos=True) for target_text in targets]

    # Dynamically pad sequences
    input_encodings = dynamic_pad_sequences(input_encodings, tokenizer.pad_id, max_length)
    target_encodings = dynamic_pad_sequences(target_encodings, tokenizer.pad_id, max_length)

    # Convert lists to tensors and move to the specified device
    input_encodings = torch.tensor(input_encodings).to(device)
    target_encodings = torch.tensor(target_encodings).to(device)

    return input_encodings, target_encodings

def main(args):
    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the datasets
    datasets = [load_dataset(dataset_name) for dataset_name in args.datasets]
    validation_dataset = load_dataset(args.validation_dataset)

    # Initialize the tokenizer and model
    tokenizer_model_path = 'vishwamai/tokenizer/tokenizer.model'
    if not os.path.isfile(tokenizer_model_path):
        raise FileNotFoundError(f"Tokenizer model file not found: {tokenizer_model_path}")

    tokenizer = Tokenizer(tokenizer_model_path)
    config = vishwamai_config.get_model_config(args.model_size)
    model = VishwamaiForCausalLM(config).to(device)

    # Preprocess the data for all tasks
    tasks = [{'inputs': dataset['train']['input'], 'targets': dataset['train']['target']} for dataset in datasets]
    input_encodings, target_encodings = preprocess_data(tasks, tokenizer, args.max_length)

    # Preprocess the validation data
    val_input_encodings, val_target_encodings = preprocess_validation_data(validation_dataset, tokenizer, args.max_length, device)

    # Define the optimizer and loss function
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _ = load_checkpoint(model, optimizer, args.checkpoint_path, device)
        model.to(device)  # Ensure the model is on the correct device after loading the checkpoint

    # Training loop
    model.train()
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        for i in range(0, len(input_encodings), args.batch_size):
            input_ids = input_encodings[i:i+args.batch_size].to(device)
            labels = target_encodings[i:i+args.batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(0, len(val_input_encodings), args.batch_size):
                val_input_ids = val_input_encodings[i:i+args.batch_size].to(device)
                val_labels = val_target_encodings[i:i+args.batch_size].to(device)

                val_outputs = model(input_ids=val_input_ids, labels=val_labels)
                val_loss += val_outputs.loss.item() * val_input_ids.size(0)

            val_loss /= len(val_input_encodings)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
                print(f"Best model saved with validation loss: {val_loss}")

        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Names of the datasets from Hugging Face datasets library")
    parser.add_argument("--validation_dataset", type=str, required=True, help="Name of the validation dataset from Hugging Face datasets library")
    parser.add_argument("--model_size", type=str, required=True, choices=['2b', '7b', '9b', '27b'], help="Size of the Vishwamai model to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for padding")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a specific checkpoint to load")
    args = parser.parse_args()
    main(args)
