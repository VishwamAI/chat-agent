import huggingface_hub

def list_datasets(author):
    search_results = huggingface_hub.list_datasets(author=author, full=True)
    for dataset_info in search_results:
        print(dataset_info.id)

if __name__ == "__main__":
    print("Datasets by Google:")
    list_datasets(author="google")

    print("\nDatasets by Microsoft:")
    list_datasets(author="microsoft")