import jax.numpy as jnp

def test_split():
    # Define the input tensor x with the same shape as in the apply_rotary_pos_emb function
    x = jnp.ones((1, 2, 8))

    # Define the split index
    split_index = 4

    # Print the shape of x before the split
    print(f"x shape before split: {x.shape}")

    # Perform the split operation
    x1, x2 = jnp.split(x, indices_or_sections=split_index, axis=-1)

    # Print the shapes of x1 and x2 after the split
    print(f"x1 shape after split: {x1.shape}")
    print(f"x2 shape after split: {x2.shape}")

if __name__ == "__main__":
    test_split()
