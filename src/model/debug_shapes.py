import jax.numpy as jnp
from architecture import apply_rotary_pos_emb

def debug_apply_rotary_pos_emb():
    # Create dummy input tensor x with shape (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = 1, 32, 32, 32
    x = jnp.ones((batch_size, num_heads, seq_len, head_dim * 2))

    # Create dummy sin and cos tensors with appropriate shapes
    sin = jnp.ones((batch_size, num_heads, seq_len, head_dim))
    cos = jnp.ones((batch_size, num_heads, seq_len, head_dim))

    # Print shapes of tensors before applying rotary position embeddings
    print(f"x shape: {x.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"cos shape: {cos.shape}")

    # Apply rotary position embeddings
    result = apply_rotary_pos_emb(x, (sin, cos))

    # Print shape of the result tensor
    print(f"result shape: {result.shape}")

if __name__ == "__main__":
    debug_apply_rotary_pos_emb()
