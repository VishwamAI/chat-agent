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
    max_length = 1024  # Maximum sequence length for the model
    for prompt in prompts:
        # Initialize conversation history with the user's prompt
        conversation_history = f"User: {prompt}\n"

        # Encode the conversation history
        input_ids = tokenizer.encode(conversation_history, return_tensors='pt')

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Truncate input_ids if it exceeds max_length
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        print("Generating response...")  # Debugging print statement
        try:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=2.0,  # Further increased repetition penalty to reduce echoing
                no_repeat_ngram_size=4  # Further increased no repeat n-gram size to reduce repetition
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during generation: {e}")
            response = "Error generating response."

        responses.append(response)
        print(f"Prompt: {prompt}")  # Debugging print statement
        print(f"Response: {response}")  # Debugging print statement
        print(f"Conversation History: {conversation_history}")  # Debugging print statement
        print(f"Input IDs: {input_ids}")  # Debugging print statement
        print(f"Output: {output}")  # Debugging print statement
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
