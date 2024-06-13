from datasets import load_dataset
from transformers import T5Tokenizer

def preprocess_chat_dataset():
    # Load the dataset
    dataset = load_dataset('ParlAI/blended_skill_talk')

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def tokenize_function(examples):
        # Tokenize the input and target text
        inputs = examples['context']
        targets = [' '.join(messages) for messages in examples['guided_messages']]  # Concatenate nested messages
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=512, truncation=True, padding=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Save the tokenized dataset
    tokenized_datasets.save_to_disk('tokenized_blended_skill_talk')

if __name__ == "__main__":
    preprocess_chat_dataset()
