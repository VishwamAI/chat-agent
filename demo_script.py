import os
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_prompts(file_path: str):
    data = pd.read_csv(file_path)
    prompts = data['prompt'].tolist()
    return prompts

def generate_responses(prompts: list, model, tokenizer):
    responses = []
    conversation_history = ""
    max_length = 1024  # Maximum sequence length for the model
    for prompt in prompts:
        conversation_history += f"User: {prompt}\n"
        input_ids = tokenizer.encode(conversation_history, return_tensors='pt')

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Truncate input_ids if it exceeds max_length
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        conversation_history += f"Bot: {response}\n"

        # Truncate conversation_history to ensure it does not exceed max_length
        conversation_history_ids = tokenizer.encode(conversation_history, return_tensors='pt')
        if conversation_history_ids.size(1) > max_length:
            conversation_history = tokenizer.decode(conversation_history_ids[0, -max_length:], skip_special_tokens=True)

        responses.append(response)
    return responses

def main():
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])

    # Load prompts from CSV
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample_dialogues.csv'))
    prompts = load_prompts(csv_file_path)

    # Generate responses
    responses = generate_responses(prompts, model, tokenizer)

    # Print responses
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()

if __name__ == "__main__":
    main()
