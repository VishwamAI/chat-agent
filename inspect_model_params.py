from src.model.architecture import VishwamAILLM
import yaml
import jax
import jax.numpy as jnp
from flax.training import checkpoints

# Load the model configuration
config_path = 'configs/default_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize the model
model = VishwamAILLM(config=config)
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
params = model.init(rng, dummy_input, is_training=False)['params']

# Restore the model state from the checkpoint
model_state = checkpoints.restore_checkpoint(ckpt_dir=config['model_name'], target=params)

# Print the model state to inspect the parameters
print(model_state)
