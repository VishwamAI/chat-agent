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
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=1.0,
            top_k=30,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
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
