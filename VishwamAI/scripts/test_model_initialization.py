import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
import logging
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

def test_model_initialization():
    example_input = "What is the capital of France?"
    tokenizer = tf_text.BertTokenizer.from_pretrained(config.MODEL_NAME)
    tokenized_input = tokenize_input(example_input, tokenizer)
    logging.info(f"Tokenized input: {tokenized_input}")
    logging.info(f"Tokenized input dtype: {tokenized_input.dtype}")

    forward, params, rng = initialize_model(tokenized_input)
    logging.info(f"Model parameters: {params}")

if __name__ == "__main__":
    test_model_initialization()
