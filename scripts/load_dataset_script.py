from datasets import load_dataset
from vishwamai.tokenizer import Tokenizer

def load_and_preprocess_dataset(dataset_name, split, tokenizer_model_path):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Initialize the tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_model_path)

    # Preprocess the dataset
    def preprocess_function(examples):
        inputs = examples["text"]
        model_inputs = tokenizer.encode(inputs, bos=True, eos=True)
        return {"input_ids": model_inputs}

    # Apply the preprocessing function to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    return tokenized_dataset

if __name__ == "__main__":
    dataset_name = "nvidia/ChatRAG-Bench"
    split = "train"
    tokenizer_model_path = "tokenizer/tokenizer.model"
    tokenized_dataset = load_and_preprocess_dataset(dataset_name, split, tokenizer_model_path)
    print(tokenized_dataset)
