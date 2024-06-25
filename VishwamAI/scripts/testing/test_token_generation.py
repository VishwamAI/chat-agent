import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
from model_architecture import VishwamAIModel

def main():
    # Initialize the model
    vocab_size = 20000
    embed_dim = 512
    num_heads = 8
    num_layers = 12
    num_experts = 4
    max_sequence_length = 1024

    model = VishwamAIModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        max_sequence_length=max_sequence_length
    )

    # Sample input prompt
    prompt = "Once upon a time"

    # Generate text
    generated_text = model.generate_text(prompt)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
