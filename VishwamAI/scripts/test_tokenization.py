import tensorflow_text as tf_text
import jax.numpy as jnp
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

def test_tokenization():
    example_input = "What is the capital of France?"
    tokenizer = tf_text.BertTokenizer.from_pretrained(config.MODEL_NAME)
    tokenized_input = tokenize_input(example_input, tokenizer)
    logging.info(f"Tokenized input: {tokenized_input}")
    logging.info(f"Tokenized input dtype: {tokenized_input.dtype}")

if __name__ == "__main__":
    test_tokenization()
