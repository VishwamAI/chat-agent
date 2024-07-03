import jax
import jax.numpy as jnp

def demonstrate_reshaping_and_broadcasting():
    # Create a dummy input tensor x with shape (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = 1, 32, 32, 32
    x = jnp.ones((batch_size, num_heads, seq_len, head_dim * 2))

    # Create dummy sin and cos tensors with appropriate shapes
    sin = jnp.ones((batch_size, num_heads, seq_len, head_dim))
    cos = jnp.ones((batch_size, num_heads, seq_len, head_dim))

    # Print shapes of tensors before applying rotary position embeddings
    print(f"x shape: {x.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"cos shape: {cos.shape}")

    # Reshape sin and cos tensors to match the shape of x1 for broadcasting
    sin = sin.reshape((batch_size, num_heads, seq_len, head_dim))
    cos = cos.reshape((batch_size, num_heads, seq_len, head_dim))

    # Print shapes of tensors after reshaping
    print(f"reshaped sin shape: {sin.shape}")
    print(f"reshaped cos shape: {cos.shape}")

    # Apply element-wise multiplication to simulate broadcasting
    x1 = x[:, :, :, :head_dim]
    x2 = x[:, :, :, head_dim:]
    x_rotated = x1 * cos + x2 * sin

    # Print shape of the result tensor
    print(f"x_rotated shape: {x_rotated.shape}")

if __name__ == "__main__":
    demonstrate_reshaping_and_broadcasting()
