import jax
import jax.numpy as jnp
from transformers import FlaxBertForSequenceClassification, AutoTokenizer
import sys
import os
import time
import pandas as pd
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import VishwamAILLM
from transformers import AutoTokenizer

def load_model(config_path, checkpoint_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = VishwamAILLM(config)
    logger.debug(f"Model initialized: {model}")

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    logger.debug(f"Shape of dummy_input: {dummy_input.shape}")
    params = model.init(rng, dummy_input)
    logger.debug(f"Parameters initialized: {params}")

    # Load trained parameters
    with open(checkpoint_path, 'rb') as f:
        trained_params = jnp.load(f, allow_pickle=True)

    # Ensure trained_params is a dictionary
    if isinstance(trained_params, dict):
        params = trained_params
    else:
        raise ValueError("Loaded parameters are not in the expected format")

    return model, params, config

def generate_and_evaluate(model, params, input_ids, config, max_length=100):
    # Ensure input_ids are JAX arrays
    input_ids = jnp.array(input_ids)
    print(f"Shape of input_ids: {input_ids.shape}")  # Debugging statement

    @jax.jit
    def generate_step(params, rng, input_ids):
        # Pass inputs through the model
        logits, _ = model.apply(params, rng, input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = jax.random.categorical(rng, next_token_logits)
        return next_token

    rng = jax.random.PRNGKey(0)  # Initialize RNG

    start_time = time.time()
    try:
        generated_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            next_token = generate_step(params, rng, generated_ids)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=-1)
    except Exception as e:
        print(f"Error during generate_step: {e}")
        raise

    end_time = time.time()
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds

    generated_text = tokenizer.decode(generated_ids[0])

    try:
        final_evaluation = model.self_evaluate(generated_text, {})
    except Exception as e:
        print(f"Error during self_evaluate: {e}")
        raise

    # Ensure unnecessary references are deleted to aid garbage collection
    del generated_ids

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
                input_ids = tokenizer.encode(input_text, return_tensors='np')  # Tokenize the current prompt and return as JAX tensor
                print(f"Shape of input_ids: {input_ids.shape}")  # Debugging statement
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

if __name__ == "__main__":
    main()
