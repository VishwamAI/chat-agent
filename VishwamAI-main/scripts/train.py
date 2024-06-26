import jax
import jax.numpy as jnp
import haiku as hk
from src.model.architecture import VishwamAILLM
from src.training.trainer import VishwamAITrainer
from src.data.dataset import create_dataset
from transformers import AutoTokenizer
import yaml

def main():
    # Load configuration
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    config['pad_token_id'] = tokenizer.pad_token_id

    # Create datasets
    train_dataset = create_dataset(config['train_file'], tokenizer, config['batch_size'], config['max_seq_length'])
    eval_dataset = create_dataset(config['eval_file'], tokenizer, config['batch_size'], config['max_seq_length'])

    # Initialize model
    def model_fn(inputs):
        model = VishwamAILLM(config)
        return model(inputs, is_training=True)

    model = hk.transform(model_fn)

    # Initialize trainer
    trainer = VishwamAITrainer(model, config)

    # Train model
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input)
    trained_params = trainer.train(params, train_dataset, eval_dataset, config['num_epochs'])

    # Save trained parameters
    # Implement saving logic here

if __name__ == "__main__":
    main()