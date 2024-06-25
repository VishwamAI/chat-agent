import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_architecture import VishwamAIModel

def forward_fn(inputs):
    # Model parameters
    vocab_size = 20000
    embed_dim = 512
    num_heads = 8
    num_layers = 12
    num_experts = 4
    max_sequence_length = 1024

    # Instantiate the model within the hk.transform context
    model = VishwamAIModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        max_sequence_length=max_sequence_length
    )

    tokenized_inputs = model.tokenizer(inputs).numpy()
    input_ids = jnp.array(tokenized_inputs, dtype=jnp.int32)
    return model(input_ids)

def main():
    # Sample input prompt
    prompt = "Once upon a time"

    # Transform the forward function
    transformed_model = hk.transform(forward_fn)

    # Generate text
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, [prompt])
    generated_text = transformed_model.apply(params, rng, [prompt])
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
