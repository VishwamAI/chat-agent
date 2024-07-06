import jax.numpy as jnp

def test_jnp_split():
    # Define the input tensor x with the same shape as in the apply_rotary_pos_emb function
    x = jnp.ones((1, 2, 8))

    # Define the split index
    split_index = 4

    # Print the shape of x before the split
    print(f"x shape before split: {x.shape}")

    # Perform the split operation
    split_result = jnp.split(x, indices_or_sections=split_index, axis=-1)

    # Print the shapes of the resulting arrays after the split
    for i, arr in enumerate(split_result):
        print(f"split_result[{i}] shape: {arr.shape}")

if __name__ == "__main__":
    test_jnp_split()
