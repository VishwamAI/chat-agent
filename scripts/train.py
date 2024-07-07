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
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architecture import ImprovedAttention  # Import ImprovedAttention from architecture.py

# Load configuration
with open('/home/ubuntu/chat-agent/configs/default_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
    model = ImprovedAttention(config)  # Use the imported ImprovedAttention class

    # Train model
    train(model, train_data, config)

if __name__ == "__main__":
    main()
