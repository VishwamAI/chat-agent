from datasets import load_dataset

# Load the English-only variant of the C4 dataset in streaming mode
en = load_dataset("allenai/c4", "en", streaming=True)

# Print a sample from the dataset to verify loading
for sample in en:
    print(sample)
    break
