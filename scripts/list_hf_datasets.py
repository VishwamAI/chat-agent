from datasets import list_datasets

def list_all_datasets():
    try:
        # List all available datasets
        datasets = list_datasets()
        print(f"Total datasets available: {len(datasets)}")
        for dataset in datasets:
            print(dataset)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_all_datasets()
