import json

def clean_dataset():
    try:
        with open('dataset.json', 'r') as f:
            data = json.load(f)

        # Filter out nonsensical or repetitive input-output pairs
        cleaned_data = [pair for pair in data if not pair['input'].startswith('Todayby')]

        with open('dataset.json', 'w') as f:
            json.dump(cleaned_data, f, indent=4)

        print("Dataset cleaned successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_dataset()
