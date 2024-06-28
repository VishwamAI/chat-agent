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

    # Load trained parameters
    with open(checkpoint_path, 'rb') as f:
        trained_params = jnp.load(f, allow_pickle=True)

    # Ensure trained_params is a dictionary
    if isinstance(trained_params, dict):
        params = trained_params
    else:
        # Convert the loaded parameters to a Haiku Params object if they are not already in that format
        if isinstance(trained_params, jnp.ndarray):
            params = hk.data_structures.to_immutable_dict(hk.data_structures.to_mutable_dict({'params': trained_params}))
        else:
            raise ValueError("Loaded parameters are not in the expected format")

    return model, params, config

def generate_and_evaluate(model, params, tokenizer, input_text, config, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='jax')

    @jax.jit
    def generate_step(params, rng, input_ids):
        return model.apply(params, rng, input_ids, method=VishwamAILLM.generate_with_evaluation)

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
        final_evaluation = model.apply(params, rng, generated_text, evaluation_metrics, method=VishwamAILLM.self_evaluate)
    except Exception as e:
        print(f"Error during self_evaluate: {e}")
        raise

    return generated_text, final_evaluation, response_time

def main():
    config_path = '/home/ubuntu/chat-agent/VishwamAI-main/configs/default_config.yaml'
    checkpoint_path = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints/model_checkpoint.npy'  # Updated to match the expected checkpoint file name

    model, params, config = load_model(config_path, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    # Log the type and structure of params
    print(f"Type of params: {type(params)}")
    if isinstance(params, dict):
        print(f"Keys in params: {list(params.keys())}")

    # Load prompts from the CSV file
    prompts = []
    train_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    with open(train_file_path, 'r') as csvfile:
        reader = pd.read_csv(csvfile)
        prompts = reader['prompt'].tolist()

    iterations = len(prompts)  # Number of iterations based on the number of prompts

    # Open a file to log training loss
    loss_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/training_loss.log'))
    with open(loss_log_file, 'w') as log_file:
        log_file.write("Iteration,Input,Generated Text,Self-evaluation,Response Time (ms)\n")

        for i in range(iterations):
            input_text = prompts[i]
            try:
                generated_text, evaluation, response_time = generate_and_evaluate(model, params, tokenizer, input_text, config)
            except Exception as e:
                print(f"Error during generate_and_evaluate: {e}")
                continue
            log_file.write(f"{i + 1},{input_text},{generated_text},{evaluation},{response_time:.2f}\n")
            print(f"Iteration {i + 1}:")
            print(f"Input: {input_text}")
            print(f"Generated text: {generated_text}")
            print(f"Self-evaluation: {evaluation}")
            print(f"Response time: {response_time:.2f} ms")

if __name__ == "__main__":
    main()
