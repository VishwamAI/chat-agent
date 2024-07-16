# Script to download selected datasets from Hugging Face
import datasets

def download_dataset(dataset_name, dataset_config=None):
    # Load the dataset with an optional config
    if dataset_config:
        dataset = datasets.load_dataset(dataset_name, dataset_config)
    else:
        dataset = datasets.load_dataset(dataset_name)
    # Save the dataset to disk
    dataset.save_to_disk(f'/home/ubuntu/chat-agent/datasets/{dataset_name}')

if __name__ == "__main__":
    # List of selected datasets to download with optional configs
    selected_datasets = [
        ('google/boolq', None),
        ('google/wiki40b', 'en'),
        ('microsoft/ms_marco', None),
        ('microsoft/orca-math-word-problems-200k', None)
    ]

    # Download each selected dataset
    for dataset_name, dataset_config in selected_datasets:
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name, dataset_config)
        print(f"Finished downloading {dataset_name}.")