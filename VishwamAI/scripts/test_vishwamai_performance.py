import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import keras_nlp
import logging
import config
import pickle
from model_architecture import VishwamAIModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tokenize_input(input_text, tokenizer):
    try:
        tokenized_input = tokenizer.tokenize(input_text)
        return jax.numpy.array(tokenized_input, dtype=jnp.int32)  # Ensure inputs are integer dtype for embedding layer
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

def load_model_params():
    try:
        with open("vishwamai_model_params.pkl", "rb") as f:
            params = pickle.load(f)
        return params
    except Exception as e:
        logging.error(f"Error loading model parameters: {e}")
        raise

def decode_output(output, tokenizer):
    # Implement decoding logic to convert model output to human-readable format
    token_ids = tf.argmax(output, axis=-1).numpy()
    decoded_output = tokenizer.detokenize(token_ids)
    return decoded_output

def test_vishwamai_performance():
    tasks = [
        {"input": "Solve the equation: 2x + 3 = 7", "expected_output": "x = 2"},
        {"input": "What is the capital of France?", "expected_output": "Paris"},
        {"input": "Explain the theory of relativity.", "expected_output": "The theory of relativity, developed by Albert Einstein, includes the special relativity and general relativity theories."},
        {"input": "What is the square root of 144?", "expected_output": "12"},
        {"input": "Who wrote 'To Kill a Mockingbird'?", "expected_output": "Harper Lee"},
        # Additional tasks for MMLU and MATH reasoning benchmarks
        {"input": "Integrate the function f(x) = x^2 from 0 to 1.", "expected_output": "1/3"},
        {"input": "Differentiate the function f(x) = sin(x).", "expected_output": "cos(x)"},
        {"input": "What is the derivative of f(x) = e^x?", "expected_output": "e^x"},
        {"input": "Solve the system of equations: 2x + y = 5 and x - y = 1.", "expected_output": "x = 2, y = 1"},
        {"input": "What is the value of pi to 5 decimal places?", "expected_output": "3.14159"}
    ]

    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=config.VOCAB_FILE)

    for task in tasks:
        input_text = task["input"]
        expected_output = task["expected_output"]

        tokenized_input = tokenize_input(input_text, tokenizer)
        logging.info(f"Tokenized input: {tokenized_input}")
        logging.info(f"Tokenized input dtype: {tokenized_input.dtype}")

        forward, _, rng = initialize_model(tokenized_input)
        params = load_model_params()
        output = process_input(forward, params, rng, tokenized_input)
        decoded_output = decode_output(output, tokenizer)
        logging.info(f"Model output: {decoded_output}")

        # Compare model output with expected output
        if decoded_output == expected_output:
            logging.info(f"Task '{input_text}' passed.")
        else:
            logging.error(f"Task '{input_text}' failed. Expected: {expected_output}, but got: {decoded_output}")

if __name__ == "__main__":
    test_vishwamai_performance()
