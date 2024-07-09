import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import json
import argparse

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data, tokenizer, max_length=512):
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    input_encodings = tokenizer(inputs, max_length=max_length, truncation=True, padding=True, return_tensors='tf')
    target_encodings = tokenizer(targets, max_length=max_length, truncation=True, padding=True, return_tensors='tf')
    return input_encodings, target_encodings

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

    # Preprocess the data
    input_encodings, target_encodings = preprocess_data(dataset, tokenizer)

    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Train the model
    model.fit(
        [input_encodings.input_ids, input_encodings.attention_mask],
        target_encodings.input_ids,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (JSON format)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()
    main(args)
