import jax
import jax.numpy as jnp
import haiku as hk
import sys
import os
import time
import pandas as pd

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import VishwamAILLM
from transformers import AutoTokenizer
import yaml

def load_model(config_path, checkpoint_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    def model_fn(inputs):
        model = VishwamAILLM(config)
        return model(inputs)

    model = hk.transform(model_fn)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    print(f"Shape of dummy_input: {dummy_input.shape}")
    params = model.init(rng, dummy_input)
    print(f"Type of initialized params: {type(params)}")
    if isinstance(params, dict):
        for key, value in params.items():
            print(f"Key: {key}, Shape: {value.shape}")
    elif isinstance(params, jnp.ndarray):
        print(f"Initialized params shape: {params.shape}")

    # Load trained parameters
    with open(checkpoint_path, 'rb') as f:
        trained_params = jnp.load(f, allow_pickle=True)

    # Log the type and structure of the loaded parameters
    print(f"Loaded trained parameters type: {type(trained_params)}")
    if isinstance(trained_params, dict):
        for key, value in trained_params.items():
            print(f"Key: {key}, Shape: {value.shape}")
    elif isinstance(trained_params, jnp.ndarray):
        print(f"Trained parameters shape: {trained_params.shape}")

    # Ensure trained_params is a dictionary
    if isinstance(trained_params, dict):
        params = trained_params
    else:
        # Convert the loaded parameters to a Haiku Params object if they are not already in that format
        if isinstance(trained_params, jnp.ndarray):
            params = hk.data_structures.to_immutable_dict(hk.data_structures.to_mutable_dict({'params': trained_params}))
        else:
            raise ValueError("Loaded parameters are not in the expected format")

    # Log the structure and dimensions of the converted parameters
    print(f"Converted parameters type: {type(params)}")
    if isinstance(params, dict):
        for key, value in params.items():
            print(f"Key: {key}, Shape: {value.shape}")

    # Debugging statement to log the mask shape
    dummy_mask = model.apply(params, rng, dummy_input)[1][0]['k']
    print(f"Shape of dummy_mask: {dummy_mask.shape}")

    return model, params, config

def generate_and_evaluate(model, params, input_ids, config, max_length=100):
    @jax.jit
    def generate_step(params, rng, input_ids):
        print(f"Shape of input_ids: {input_ids.shape}")  # Debugging statement
        output = model.apply(params, rng, input_ids)
        print(f"Shape of output: {output[0].shape}")  # Debugging statement
        print(f"Shape of attn before matmul: {output[1]['attn'].shape}")  # Debugging statement
        print(f"Shape of v before matmul: {output[1]['v'].shape}")  # Debugging statement
        return output

    rng = jax.random.PRNGKey(0)  # Initialize RNG

    start_time = time.time()
    try:
        generated_ids, evaluation_metrics = generate_step(params, rng, input_ids)
    except Exception as e:
        print(f"Error during generate_step: {e}")
        raise

    end_time = time.time()
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds

    generated_text = tokenizer.decode(generated_ids[0])

    try:
        final_evaluation = VishwamAILLM.self_evaluate(generated_text, evaluation_metrics)
    except Exception as e:
        print(f"Error during self_evaluate: {e}")
        raise

    # Ensure unnecessary references are deleted to aid garbage collection
    del generated_ids, evaluation_metrics

    return generated_text, final_evaluation, response_time

def main():
    config_path = '/home/ubuntu/chat-agent/configs/default_config.yaml'
    checkpoint_path = '/home/ubuntu/chat-agent/checkpoints/model_checkpoint.npy'  # Updated to match the expected checkpoint file name

    model, params, config = load_model(config_path, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    # Log the type and structure of params
    print(f"Type of params: {type(params)}")
    if isinstance(params, dict):
        print(f"Keys in params: {list(params.keys())}")

    train_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    with open(train_file_path, 'r') as csvfile:
        reader = pd.read_csv(csvfile)
        iterations = len(reader)  # Number of iterations based on the number of prompts

    # Open a file to log training loss
    loss_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/training_loss.log'))
    with open(loss_log_file, 'w') as log_file:
        log_file.write("Iteration,Input,Generated Text,Self-evaluation,Response Time (ms)\n")

        with open(train_file_path, 'r') as csvfile:
            reader = pd.read_csv(csvfile)
            for i, row in reader.iterrows():
                input_text = row['prompt']
                input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize the current prompt and return as PyTorch tensor
                try:
                    generated_text, evaluation, response_time = generate_and_evaluate(model, params, input_ids, config)
                except Exception as e:
                    print(f"Error during generate_and_evaluate: {e}")
                    continue
                log_file.write(f"{i + 1},{input_text},{generated_text},{evaluation},{response_time:.2f}\n")
                print(f"Iteration {i + 1}:")
                print(f"Input: {input_text}")
                print(f"Generated text: {generated_text}")
                print(f"Self-evaluation: {evaluation}")
                print(f"Response time: {response_time:.2f} ms")

                # Ensure unnecessary references are deleted to aid garbage collection
                del input_ids, generated_text, evaluation, response_time

                # Explicitly call garbage collection
                import gc
                gc.collect()

if __name__ == "__main__":
    main()
