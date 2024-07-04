import os
import pandas as pd
import yaml
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from flax.training import checkpoints
from src.model.architecture import VishwamAILLM

def load_prompts(file_path: str):
    data = pd.read_csv(file_path)
    prompts = data['prompt'].tolist()
    return prompts

def generate_responses(prompts: list, model, tokenizer):
    responses = []
    max_length = 256  # Maximum sequence length for the model
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
        input_ids = tokenizer.encode(conversation_history_str, return_tensors='np')

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Truncate input_ids if it exceeds max_length
        if (input_ids.shape[1] > max_length):
            input_ids = input_ids[:, -max_length:]

        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).astype(int)

        try:
            # Ensure input_ids has the correct shape
            if len(input_ids.shape) == 2:
                input_ids = input_ids[:, :, None]  # Add a third dimension if input_ids is two-dimensional

            # Ensure input_ids has the correct number of elements
            batch_size, seq_len, embed_dim = input_ids.shape
            expected_embed_dim = model.config['num_heads'] * model.config['head_dim']
            if embed_dim != expected_embed_dim:
                if embed_dim * seq_len == expected_embed_dim:
                    input_ids = input_ids.reshape(batch_size, seq_len, model.config['num_heads'], model.config['head_dim'])
                else:
                    raise ValueError(f"Cannot reshape array of shape {input_ids.shape} to (batch_size, seq_len, num_heads, head_dim)")

            output, _ = model.apply({'params': model.params}, input_ids, is_training=False)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"

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

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    # Initialize model
    model = VishwamAILLM(config=config)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']
    model_state = checkpoints.restore_checkpoint(ckpt_dir=config['model_name'], target=params)
    model = model.replace(params=model_state)

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
