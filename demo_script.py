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

        # Maintain a sliding window of the last 5 exchanges (user prompt + bot response)
        if (len(conversation_history) > 10):
            conversation_history = conversation_history[-10:]

        # Use the full conversation history for generating the next response
        conversation_history_str = " ".join(conversation_history)

        # Encode the conversation history
        input_ids = tokenizer.encode(conversation_history_str, return_tensors='pt')

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
                max_new_tokens=150,  # Increased max_new_tokens for longer responses
                temperature=0.9,  # Adjusted temperature for more coherent responses
                top_k=50,  # Adjusted top_k for more diverse responses
                top_p=0.95,  # Adjusted top_p for more diverse responses
                do_sample=True,
                repetition_penalty=1.2,  # Adjusted repetition penalty
                no_repeat_ngram_size=3,  # Adjusted no repeat n-gram size
                num_beams=1,  # Simplified to no beam search
                num_return_sequences=1  # Return only one sequence
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"

        # Check if the response is echoing the prompt or conversation history
        if response.strip().lower() in [prompt.strip().lower(), conversation_history_str.strip().lower()]:
            response = "I'm sorry, I didn't understand that. Can you please rephrase?"

        # Append the bot's response to the conversation history
        conversation_history.append(response)

        # Append the bot's response to the responses list
        responses.append(response)

        # Print the prompt and response for verification
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Input IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")
        print(f"Output: {output}")
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
