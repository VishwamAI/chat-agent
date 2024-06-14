import csv
import os
import pickle
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def preprocess_csv(file_path):
    """
    Preprocess a CSV file containing questions and answer choices.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: A list of tuples containing tokenized inputs and labels.
    """
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                question = row[0]
                choices = row[1:5]
                correct_answer = row[5]
                input_text = f"Question: {question} Choices: {choices}"
                inputs = tokenizer(input_text, return_tensors="pt")
                label = choices.index(correct_answer)
                data.append((inputs, label))
            except IndexError:
                print(f"Skipping malformed row in file {file_path}: {row}")
            except ValueError:
                print(f"Skipping row with invalid correct answer in file {file_path}: {row}")
    return data

def preprocess_dataset(data_dir):
    """
    Preprocess all CSV files in a directory.

    Args:
        data_dir (str): Path to the directory containing CSV files.

    Returns:
        list: A list of tuples containing tokenized inputs and labels from all CSV files.
    """
    dataset = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                dataset.extend(preprocess_csv(file_path))
    return dataset

def save_preprocessed_data(data, output_file):
    """
    Save preprocessed data to a file.

    Args:
        data (list): Preprocessed data to be saved.
        output_file (str): Path to the output file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    data_dir = "../data"
    dataset = preprocess_dataset(data_dir)
    print(f"Preprocessed {len(dataset)} examples.")

    # Split the dataset into training, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Save the preprocessed data to disk
    save_preprocessed_data(train_data, "../data/train_data.pkl")
    save_preprocessed_data(val_data, "../data/val_data.pkl")
    save_preprocessed_data(test_data, "../data/test_data.pkl")
