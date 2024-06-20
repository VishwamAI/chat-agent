import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
import logging
import config
from model_architecture import VishwamAIModel

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

def test_vishwamai_performance():
    tasks = [
        {"input": "Solve the equation: 2x + 3 = 7", "expected_output": "x = 2"},
        {"input": "What is the capital of France?", "expected_output": "Paris"},
        {"input": "Explain the theory of relativity.", "expected_output": "The theory of relativity, developed by Albert Einstein, includes the special relativity and general relativity theories."},
        {"input": "What is the square root of 144?", "expected_output": "12"},
        {"input": "Who wrote 'To Kill a Mockingbird'?", "expected_output": "Harper Lee"}
    ]

    vocab_lookup_table = tf_text.vocab_lookup_table_from_file(config.VOCAB_FILE)
    tokenizer = tf_text.BertTokenizer(vocab_lookup_table)

    for task in tasks:
        input_text = task["input"]
        expected_output = task["expected_output"]

        tokenized_input = tokenize_input(input_text, tokenizer)
        logging.info(f"Tokenized input: {tokenized_input}")
        logging.info(f"Tokenized input dtype: {tokenized_input.dtype}")

        forward, params, rng = initialize_model(tokenized_input)
        output = process_input(forward, params, rng, tokenized_input)
        logging.info(f"Model output: {output}")

        # Compare model output with expected output
        if output == expected_output:
            logging.info(f"Task '{input_text}' passed.")
        else:
            logging.error(f"Task '{input_text}' failed. Expected: {expected_output}, but got: {output}")

if __name__ == "__main__":
    test_vishwamai_performance()
