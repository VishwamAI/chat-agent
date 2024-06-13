from transformers import T5Tokenizer

def create_custom_dataset():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Define custom input-output pairs
    custom_data = [
        {"input_text": "hi", "output_text": "hello how can I assist you today"},
        {"input_text": "hello", "output_text": "hi there, how can I help you"},
        {"input_text": "good morning", "output_text": "good morning, how can I assist you"},
        {"input_text": "good evening", "output_text": "good evening, how can I help you"},
        {"input_text": "how are you", "output_text": "I'm an AI, so I don't have feelings, but I'm here to help you"},
        {"input_text": "what's your name", "output_text": "I am the Vishwam model, your AI assistant"},
        {"input_text": "thank you", "output_text": "You're welcome! How can I assist you further"},
        {"input_text": "bye", "output_text": "Goodbye! Have a great day"}
    ]

    # Tokenize the custom data
    tokenized_data = []
    for pair in custom_data:
        input_ids = tokenizer(pair["input_text"], padding="max_length", truncation=True, max_length=512)['input_ids']
        labels = tokenizer(pair["output_text"], padding="max_length", truncation=True, max_length=512)['input_ids']
        tokenized_data.append({"input_ids": input_ids, "labels": labels})

    return tokenized_data

if __name__ == "__main__":
    custom_dataset = create_custom_dataset()
    for data in custom_dataset:
        print(data)
