from datasets import list_datasets

def main():
    # List all available datasets
    datasets = list_datasets()
    print("Available datasets on Hugging Face:")
    for dataset in datasets:
        print(dataset)

if __name__ == "__main__":
    main()
