import flax.linen as nn

class VishwamAILLM(nn.Module):
    config: dict

    def setup(self):
        self.embed_dim = self.config['embed_dim']
        self.num_heads = self.config['num_heads']
        self.head_dim = self.config['head_dim']
        self.qkv_dense = nn.Dense(self.embed_dim * 3)
        self.out_dense = nn.Dense(self.embed_dim)

    def __call__(self, input_ids, attention_mask=None, is_training=False):
        # Debugging: Print the shape and values of attention_mask at the start of __call__
        print(f"attention_mask shape at start of __call__: {attention_mask.shape}")
        print(f"attention_mask values at start of __call__: {attention_mask}")

        # Ensure attention_mask is a 2D tensor
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(f"Attention mask is not a 2D tensor: {attention_mask.shape}")

        # Additional debugging: Print the shape and values of attention_mask before any operations
        print(f"attention_mask shape before operations: {attention_mask.shape}")
        print(f"attention_mask values before operations: {attention_mask}")

        # Apply the attention mechanism
        qkv = self.qkv_dense(input_ids)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Additional debugging: Print the shape and values of q, k, v
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")

        # Compute attention scores
        attn_scores = jnp.einsum('...qd,...kd->...qk', q, k) / jnp.sqrt(self.head_dim)

        # Additional debugging: Print the shape and values of attn_scores before applying attention_mask
        print(f"attn_scores shape before applying attention_mask: {attn_scores.shape}")
        print(f"attn_scores values before applying attention_mask: {attn_scores}")

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Additional debugging: Print the shape and values of attn_scores after applying attention_mask
        print(f"attn_scores shape after applying attention_mask: {attn_scores.shape}")
        print(f"attn_scores values after applying attention_mask: {attn_scores}")

        # Compute attention weights
        attn_weights = nn.softmax(attn_scores, axis=-1)

        # Compute attention output
        attn_output = jnp.einsum('...qk,...vd->...qd', attn_weights, v)

        # Apply output dense layer
        output = self.out_dense(attn_output)

        # Additional debugging: Print the shape and values of output
        print(f"output shape: {output.shape}")
        print(f"output values: {output}")

        return output
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
        print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

        # Ensure input_ids is a 2D tensor
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids is not a 2D tensor: {input_ids.shape}")

        # Debugging: Print the values of input_ids and tokenizer.pad_token_id before comparison
        print(f"input_ids values before comparison: {input_ids}")
        print(f"tokenizer.pad_token_id value: {tokenizer.pad_token_id}")

        # Ensure input_ids and tokenizer.pad_token_id are correctly defined
        if input_ids.size == 0 or tokenizer.pad_token_id is None:
            raise ValueError("input_ids or tokenizer.pad_token_id is not correctly defined")

        # Correctly create the attention mask as a 2D tensor
        attention_mask = (input_ids != tokenizer.pad_token_id).astype(jnp.float32)

        # Debugging: Print the shape and values of attention_mask after creation
        print(f"attention_mask shape after creation: {attention_mask.shape}")
        print(f"attention_mask values after creation: {attention_mask}")

        # Ensure attention_mask is a 2D tensor
        if attention_mask.ndim != 2:
            raise ValueError(f"Attention mask is not a 2D tensor: {attention_mask.shape}")

        # Ensure attention_mask is not empty and has the correct shape
        if attention_mask.size == 0:
            raise ValueError("attention_mask is empty after creation")
        if attention_mask.shape != (input_ids.shape[0], input_ids.shape[1]):
            raise ValueError(f"Attention mask shape mismatch: expected {(input_ids.shape[0], input_ids.shape[1])}, but got {attention_mask.shape}")

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

            # Ensure attention_mask is correctly shaped before passing to the model
            if attention_mask.shape != (input_ids.shape[0], input_ids.shape[1]):
                raise ValueError(f"Attention mask shape mismatch before model: expected {(input_ids.shape[0], input_ids.shape[1])}, but got {attention_mask.shape}")

            # Additional debugging: Print the shape of attention_mask immediately before model.apply
            print(f"attention_mask shape immediately before model.apply: {attention_mask.shape}")
            print(f"attention_mask values immediately before model.apply: {attention_mask}")
            print(f"attention_mask dtype immediately before model.apply: {attention_mask.dtype}")

            # Additional debugging: Print the type of attention_mask immediately before model.apply
            print(f"attention_mask type immediately before model.apply: {type(attention_mask)}")

            # Ensure attention_mask is a 2D tensor
            if attention_mask.ndim != 2:
                raise ValueError(f"Attention mask is not a 2D tensor: {attention_mask.shape}")

            # Additional debugging: Print the shape and values of attention_mask right before model.apply
            print(f"attention_mask shape right before model.apply: {attention_mask.shape}")
            print(f"attention_mask values right before model.apply: {attention_mask}")

            # Explicitly reshape attention_mask to ensure it is a 2D tensor
            attention_mask = attention_mask.reshape((input_ids.shape[0], input_ids.shape[1]))

            # Additional debugging: Print the shape, values, dtype, and type of attention_mask after reshaping
            print(f"attention_mask shape after reshaping: {attention_mask.shape}")
            print(f"attention_mask values after reshaping: {attention_mask}")
            print(f"attention_mask dtype after reshaping: {attention_mask.dtype}")
            print(f"attention_mask type after reshaping: {type(attention_mask)}")

            # Ensure attention_mask is correctly shaped before passing to the model
            if attention_mask.shape != (input_ids.shape[0], input_ids.shape[1]):
                raise ValueError(f"Attention mask shape mismatch before model: expected {(input_ids.shape[0], input_ids.shape[1])}, but got {attention_mask.shape}")

            output = model.apply({'params': model.params}, input_ids, is_training=False, attention_mask=attention_mask)

            # Additional debugging: Print the shape and values of output after model.apply
            print(f"output shape after model.apply: {output.shape}")
            print(f"output values after model.apply: {output}")

            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            print(f"Exception: {str(e)}")
            print(f"input_ids shape at exception: {input_ids.shape}")
            print(f"attention_mask shape at exception: {attention_mask}")
            print(f"input_ids values at exception: {input_ids}")
            print(f"attention_mask values at exception: {attention_mask}")

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
