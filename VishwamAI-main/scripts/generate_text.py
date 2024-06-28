import jax
import jax.numpy as jnp
import haiku as hk
import sys
import os

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
        return model(inputs, is_training=False)

    model = hk.transform(model_fn)

    # Load trained parameters
    with open(checkpoint_path, 'rb') as f:
        trained_params = jnp.load(f)

    return model, trained_params, config

def generate_and_evaluate(model, params, tokenizer, input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='jnp')

    @jax.jit
    def generate_step(params, input_ids):
        return model.apply(params, None, input_ids, method=VishwamAILLM.generate_with_evaluation)

    generated_ids, evaluation_metrics = generate_step(params, input_ids)
    generated_text = tokenizer.decode(generated_ids[0])

    final_evaluation = model.apply(params, None, generated_text, evaluation_metrics, method=VishwamAILLM.self_evaluate)

    return generated_text, final_evaluation

def main():
    config_path = '/home/ubuntu/chat-agent/VishwamAI-main/configs/default_config.yaml'
    checkpoint_path = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints/model_checkpoint.npy'  # Updated to match the expected checkpoint file name

    model, params, config = load_model(config_path, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    input_text = "Hi"
    iterations = 5  # Number of iterations for self-reinforcement

    for i in range(iterations):
        generated_text, evaluation = generate_and_evaluate(model, params, tokenizer, input_text)
        print(f"Iteration {i + 1}:")
        print(f"Input: {input_text}")
        print(f"Generated text: {generated_text}")
        print(f"Self-evaluation: {evaluation}")
        input_text = generated_text  # Use the generated text as the new input

if __name__ == "__main__":
    main()
