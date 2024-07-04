import os
import time
import gc
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.example_libraries import optimizers
from haiku import PRNGSequence
import haiku as hk
import optax
import yaml
from typing import Iterable

# Load configuration
with open('/home/ubuntu/chat-agent/configs/default_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define the model architecture
class ImprovedAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: int, w_init_scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init_scale = w_init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Reshape input tensor for multi-head attention
        batch_size, seq_length, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        if embed_dim != self.num_heads * head_dim:
            raise ValueError(f"Embedding dimension {embed_dim} is not divisible by the number of heads {self.num_heads}")
        x = x.reshape(batch_size, seq_length, self.num_heads, head_dim)
        return x

# Define the training loop
def train(model, data, config):
    # Initialize optimizer
    opt_init, opt_update, get_params = optimizers.adam(config['learning_rate'])
    opt_state = opt_init(model)

    @jit
    def update(step, opt_state, batch):
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(model)(params, batch)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, loss

    for epoch in range(config['num_epochs']):
        for batch in data:
            opt_state, loss = update(epoch, opt_state, batch)
            print(f"Epoch {epoch}, Loss: {loss}")

        # Save checkpoint
        save_checkpoint(model, config['model_name'], step=epoch)

# Define the checkpoint saving function
def save_checkpoint(model, checkpoint_path, step):
    params = model.params
    checkpoint = {'params': params, 'step': step}
    with open(checkpoint_path, 'wb') as f:
        np.save(f, checkpoint)

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

# Main function
def main():
    # Load data
    train_data = load_data(config['train_file'])
    eval_data = load_data(config['eval_file'])

    # Initialize model
    model = ImprovedAttention(
        num_heads=config['num_heads'],
        key_size=config['embed_dim'] // config['num_heads'],
        w_init_scale=1.0
    )

    # Train model
    train(model, train_data, config)

if __name__ == "__main__":
    main()
