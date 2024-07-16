# Script to download selected datasets from Hugging Face
import datasets

def download_dataset(dataset_name):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)
    # Save the dataset to disk
    dataset.save_to_disk(f'/home/ubuntu/chat-agent/datasets/{dataset_name}')

if __name__ == "__main__":
    # List of selected datasets to download
    selected_datasets = [
        'google/boolq',
        'google/wiki40b',
        'microsoft/ms_marco',
        'microsoft/orca-math-word-problems-200k'
    ]

    # Download each selected dataset
    for dataset_name in selected_datasets:
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name)
        print(f"Finished downloading {dataset_name}.")