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

        # Debugging: Print the shape and values of input_ids after encoding
        print(f"input_ids shape after encoding: {input_ids.shape}")
        print(f"input_ids values after encoding: {input_ids}")

        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Debugging: Print the pad_token_id
        print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

        # Truncate input_ids if it exceeds max_length
        if (input_ids.shape[1] > max_length):
            input_ids = input_ids[:, -max_length:]

        # Debugging: Print the shape and values of input_ids after truncation
        print(f"input_ids shape after truncation: {input_ids.shape}")
        print(f"input_ids values after truncation: {input_ids}")

        # Ensure input_ids is not empty
        if input_ids.size == 0:
            raise ValueError("input_ids is empty after encoding")

        # Create attention mask
        print(f"input_ids before attention_mask creation: {input_ids}")
        attention_mask = (input_ids != tokenizer.pad_token_id).astype(jnp.float32)
        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))  # Expand dimensions to [batch_size, 1, 1, sequence_length]
        attention_mask = jnp.broadcast_to(attention_mask, (input_ids.shape[0], 1, 1, input_ids.shape[1]))  # Broadcast to [batch_size, 1, 1, sequence_length]

        # Debugging: Print the shape and values of attention_mask after creation
        print(f"attention_mask shape after creation: {attention_mask.shape}")
        print(f"attention_mask values after creation: {attention_mask}")

        # Ensure attention_mask is not empty and has the correct shape
        if attention_mask.size == 0:
            raise ValueError("attention_mask is empty after creation")
        if attention_mask.shape != (input_ids.shape[0], 1, 1, input_ids.shape[1]):
            raise ValueError(f"Attention mask shape mismatch: expected {(input_ids.shape[0], 1, 1, input_ids.shape[1])}, but got {attention_mask.shape}")

        # Debugging: Print the shape and values of input_ids and attention_mask
        print(f"input_ids shape: {input_ids.shape}")
        print(f"input_ids values: {input_ids}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"attention_mask values: {attention_mask}")

        try:
            # Debugging: Print the shape of input_ids and attention_mask before passing to the model
            print(f"input_ids shape before model: {input_ids.shape}")
            print(f"attention_mask shape before model: {attention_mask.shape}")
            print(f"input_ids values before model: {input_ids}")
            print(f"attention_mask values before model: {attention_mask}")

            output = model.apply({'params': model.params}, input_ids, is_training=False, attention_mask=attention_mask)
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

    # Ensure embed_dim matches num_heads * head_dim
    expected_embed_dim = config['num_heads'] * config['head_dim']
    if config['embed_dim'] != expected_embed_dim:
        raise ValueError(f"Configuration error: embed_dim {config['embed_dim']} does not match num_heads * head_dim {expected_embed_dim}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    # Initialize model
    model = VishwamAILLM(config=config)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length'], config['embed_dim']), dtype=jnp.float32)

    # Debugging: Print the shape and values of dummy_input
    print(f"dummy_input shape: {dummy_input.shape}")
    print(f"dummy_input values: {dummy_input}")

    params = model.init(rng, dummy_input)['params']
    model_state = checkpoints.restore_checkpoint(ckpt_dir=config['model_name'], target=params)
    model = model.replace(params=model_state)

    # Debugging: Print the model parameters after initialization
    print(f"Model parameters after initialization: {model.params}")

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
