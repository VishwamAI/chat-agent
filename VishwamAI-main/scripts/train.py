import sys
import os
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import yaml
import pandas as pd
import logging
from transformers import AutoTokenizer
from bias_analysis import analyze_bias
from generate_modular_question import generate_modular_question
from typing import Iterable
import more_itertools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import VishwamAILLM
from src.training.trainer import VishwamAITrainer
from src.data.dataset import create_dataset

def create_dataset_from_csv(file_path: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    def load_and_preprocess_data(file_path: str):
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data from CSV: {data.head()}")
        for _, row in data.iterrows():
            prompt = row['prompt']
            response = row['response']

            # Check for empty or None values in prompt and response
            if not prompt or not response:
                logger.warning(f"Warning: Empty or None value encountered in row: {row}")
                continue

            tokens = tokenizer.encode(prompt.strip() + " " + response.strip())
            logger.debug(f"Original tokens: {tokens}")
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))

            # Ensure pad_token_id is valid and replace None values
            if tokenizer.pad_token_id is None:
                raise ValueError("pad_token_id is None. Ensure the tokenizer is configured correctly.")
            tokens = [token if token is not None else tokenizer.pad_token_id for token in tokens]

            logger.debug(f"Padded tokens: {tokens}")
            input_ids = tokens[:-1]
            labels = tokens[1:]
            logger.debug(f"Processed input_ids: {input_ids}")
            logger.debug(f"Processed labels: {labels}")
            yield {'input_ids': input_ids, 'labels': labels}

    def create_batch(samples):
        logger.debug(f"Samples before processing: {samples}")  # Added logging
        input_ids = []
        labels = []
        for s in samples:
            if s['input_ids'] is None:
                logger.warning(f"Warning: None value encountered in input_ids: {s}")
                input_ids.append([tokenizer.pad_token_id] * max_length)
            else:
                input_ids.append(s['input_ids'])

            if s['labels'] is None:
                logger.warning(f"Warning: None value encountered in labels: {s}")
                labels.append([tokenizer.pad_token_id] * max_length)
            else:
                labels.append(s['labels'])

        logger.debug(f"Final input_ids: {input_ids}")
        logger.debug(f"Final labels: {labels}")
        batch = {
            'input_ids': jnp.array(input_ids),
            'labels': jnp.array(labels)
        }
        return batch

    dataset = load_and_preprocess_data(file_path)
    batched_dataset = (create_batch(samples) for samples in more_itertools.chunked(dataset, batch_size))
    return batched_dataset

def main():
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    # Ensure pad_token_id is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config['pad_token_id']
    else:
        config['pad_token_id'] = tokenizer.pad_token_id

    # Add mathematical symbols to tokenizer
    math_symbols = ['+', '-', '*', '/', '=', '(', ')', '^', 'sqrt', 'pi', 'e']
    tokenizer.add_tokens(math_symbols)

    # Create datasets
    train_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    eval_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    train_dataset = create_dataset_from_csv(train_file_path, tokenizer, config['batch_size'], config['max_seq_length'])
    eval_dataset = create_dataset_from_csv(eval_file_path, tokenizer, config['batch_size'], config['max_seq_length'])

    # Analyze training data for biases
    logger.info("Analyzing training data for biases...")
    for batch in train_dataset:
        text_batch = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for text in text_batch:
            bias_results = analyze_bias(text)
            logger.info(f"Bias Analysis Results for training data: {bias_results}")

    # Initialize model
    def model_fn(inputs):
        model = VishwamAILLM(config)
        return model(inputs, is_training=True)

    model = hk.transform(model_fn)

    # Initialize optimizer
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(model.init(jax.random.PRNGKey(0), jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)))

    # Initialize trainer
    trainer = VishwamAITrainer(model, config, optimizer, opt_state)

    # Train model
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input)

    logger.info("Starting training process...")
    checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        for epoch in range(config['num_epochs']):
            start_time = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
            train_loss = 0
            train_steps = 0
            for batch in train_dataset:
                batch['input_ids'] = trainer.preprocess_input(batch['input_ids'])
                batch['input_ids'] = trainer.preprocess_math_input(batch['input_ids'])
                params, trainer.opt_state, loss, _ = trainer.train_step(params, trainer.opt_state, batch)
                train_loss += loss
                train_steps += 1

                if train_steps % 100 == 0:
                    logger.info(f"Step {train_steps}: Current Train Loss: {loss:.4f}")

                # Save intermediate checkpoint every 500 steps
                if train_steps % 500 == 0:
                    intermediate_checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{train_steps}.npy')
                    np.save(intermediate_checkpoint_path, params)
                    logger.info(f"Intermediate checkpoint saved at {intermediate_checkpoint_path}")

            eval_metrics = trainer.evaluate(params, eval_dataset)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']} completed in {epoch_time:.2f} seconds")
            logger.info(f"Train Loss: {train_loss / train_steps:.4f}")
            logger.info(f"Eval Metrics: {eval_metrics}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.npy')
            np.save(checkpoint_path, params)
            logger.info(f"Checkpoint saved successfully at {checkpoint_path} after epoch {epoch + 1}")

            if trainer._should_stop_early(eval_metrics):
                logger.info("Early stopping criteria met. Ending training.")
                break

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        interrupted_checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint_interrupted.npy')
        np.save(interrupted_checkpoint_path, params)
        logger.info(f"Checkpoint saved at {interrupted_checkpoint_path} due to interruption.")

    logger.info("Training process completed.")

    # Save trained parameters
    model.save_pretrained('/home/ubuntu/chat-agent/VishwamAI-main/saved_models')
    logger.info("Trained parameters saved.")

    # Save final checkpoint
    checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint_final.npy')
    np.save(checkpoint_path, trained_params)
    logger.info(f"Final checkpoint saved at {checkpoint_path}")

    # Save checkpoint after each epoch
    for epoch in range(config['num_epochs']):
        checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.npy')
        logger.info(f"Saving checkpoint to {checkpoint_path} after epoch {epoch + 1}")
        np.save(checkpoint_path, trained_params)
        logger.info(f"Checkpoint saved successfully at {checkpoint_path} after epoch {epoch + 1}")

    # Analyze model outputs for biases
    logger.info("Analyzing model outputs for biases...")
    for batch in eval_dataset:
        text_batch = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for text in text_batch:
            bias_results = analyze_bias(text)
            logger.info(f"Bias Analysis Results for model outputs: {bias_results}")

if __name__ == "__main__":
    main()
