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
    conversation_history = []

    for prompt in prompts:
        # Add the user's prompt to the conversation history
        conversation_history.append(prompt)

        # Maintain a sliding window of the last 5 user prompts
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]

        # Encode only the last user prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Truncate input_ids if it exceeds max_length
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        try:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50,
                temperature=0.7,  # Adjusted temperature for more coherent responses
                top_k=30,  # Adjusted top_k for more focused responses
                top_p=0.85,  # Adjusted top_p for more focused responses
                do_sample=True,
                repetition_penalty=1.5,  # Increased repetition penalty
                no_repeat_ngram_size=3,  # Increased no repeat n-gram size
                num_beams=1,  # Simplified to no beam search
                num_return_sequences=1  # Return only one sequence
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"

        # Check if the response is echoing the prompt
        if response.strip().lower() == prompt.strip().lower():
            response = "I'm sorry, I didn't understand that. Can you please rephrase?"

        # Append the bot's response to the responses list only if it's a valid response
        if response != "Error generating response." and response != "I'm sorry, I didn't understand that. Can you please rephrase?":
            responses.append(response)

        # Print the prompt and response for verification
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()

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

    # Terminate the script after processing all prompts
    print("All prompts have been processed. Terminating the script.")

if __name__ == "__main__":
    main()
