from datasets import load_from_disk

def preprocess_vietnamese_chat_alpaca(dataset_path, output_file):
    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # Open the output file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through the dataset and extract conversations
        for example in dataset['train']:
            conversations = example['conversations']
            for turn in conversations:
                # Write each turn to the output file
                f.write(f"{turn['from']}: {turn['value']}\n")
            # Add a newline to separate conversations
            f.write("\n")

if __name__ == "__main__":
    dataset_path = "/home/ubuntu/chat-agent/Vietnamese-Multi-turn-Chat-Alpaca-dataset"
    output_file = "/home/ubuntu/chat-agent/data/processed/train.txt"
    preprocess_vietnamese_chat_alpaca(dataset_path, output_file)
