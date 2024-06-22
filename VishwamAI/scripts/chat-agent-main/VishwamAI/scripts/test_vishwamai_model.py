import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import keras_nlp
from model_architecture import VishwamAIModel
import config

def forward_fn(tokenized_input):
    model = VishwamAIModel()
    return model(tokenized_input)

def test_vishwamai_model():
    # Example inputs for testing
    example_inputs = [
        "What is the capital of France?",
        "Solve the equation: 2x + 3 = 7",
        "Explain the theory of relativity.",
        "What is the significance of the Renaissance period in art?",
        "Describe the process of photosynthesis."
    ]

    # Initialize the forward function with hk.transform
    forward = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)

    for input_text in example_inputs:
        # Tokenize the input
        tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=config.VOCAB_FILE)
        tokenized_input = tokenizer.tokenize(input_text)
        tokenized_input = jax.numpy.array(tokenized_input, dtype=jnp.int32)  # Ensure inputs are integer dtype for embedding layer

        # Print the data type of tokenized_input to verify conversion
        print(f"Data type of tokenized_input before forward.init: {tokenized_input.dtype}")

        # Initialize the model parameters
        params = forward.init(rng, tokenized_input)

        # Print the data type of tokenized_input after forward.init
        print(f"Data type of tokenized_input after forward.init: {tokenized_input.dtype}")

        # Process the input through the model
        output = forward.apply(params, rng, tokenized_input)

        # Print the input and output
        print(f"Input: {input_text}")
        print(f"Model output: {output}")

    # Test the self-improvement method
    model = VishwamAIModel()
    model.self_improve()

if __name__ == "__main__":
    test_vishwamai_model()
