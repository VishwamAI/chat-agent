import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import keras_nlp
import random
import logging
from model_architecture import VishwamAIModel
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tokenize_input(input_text, tokenizer):
    try:
        tokenized_input = tokenizer.tokenize(input_text).to_tensor()
        return jax.numpy.array(tokenized_input, dtype=jnp.float32)  # Ensure inputs are floating-point dtype for embedding layer
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        raise

def initialize_model(tokenized_input):
    try:
        forward = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(config.RNG_SEED)
        params = forward.init(rng, tokenized_input)
        return forward, params, rng
    except Exception as e:
        logging.error(f"Error during model initialization: {e}")
        raise

def process_input(forward, params, rng, tokenized_input):
    try:
        return forward.apply(params, rng, tokenized_input)
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        raise

def forward_fn(tokenized_input):
    model = VishwamAIModel()
    return model(tokenized_input)

def main():
    try:
        # Example input
        example_input = "What is the capital of France?"
        tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=config.VOCAB_FILE)
        tokenized_input = tokenize_input(example_input, tokenizer)

        # Debugging print statements to check dtype
        logging.info(f"Tokenized input dtype before model init: {tokenized_input.dtype}")

        # Initialize the model
        forward, params, rng = initialize_model(tokenized_input)

        # Process the input through the model
        output = process_input(forward, params, rng, tokenized_input)

        # Print the output
        logging.info(f"Model output: {output}")

        # Self-improvement example
        model = VishwamAIModel()
        model.self_improve()
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
