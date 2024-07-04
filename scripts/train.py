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
import psutil  # Import psutil module
from typing import Dict, Optional, Tuple, Iterable  # Import Dict, Optional, Tuple, and Iterable from typing module
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # Import necessary modules from transformers library

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add file handler to write logs to a file
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/train.log'))
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

import flax.linen as nn

from transformers import AutoTokenizer

def loss_fn(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

def log_memory_usage():
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.used / (1024 * 1024)  # Convert to MiB
    available_memory = memory_info.available / (1024 * 1024)  # Convert to MiB
    timestamp = time.time()
    logger.info(f"Logging memory usage: {memory_usage:.2f} MiB used, {available_memory:.2f} MiB available")
    with open(memory_log_file, 'a') as f:
        f.write(f"{timestamp},{memory_usage:.2f},{available_memory:.2f}\n")
    gc.collect()  # Explicitly call garbage collector to free up memory

def create_dataset_from_csv(file_path: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    def load_and_preprocess_data(file_path: str):
        chunk_size = 1  # Further reduce chunk size to manage memory usage
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            logger.info(f"Loaded data chunk from CSV: {chunk.head()}")
            for _, row in chunk.iterrows():
                prompt = row['prompt']
                response = row['response']

                # Check for empty or None values in prompt and response
                if not prompt or not response:
                    logger.warning(f"Warning: Empty or None value encountered in row: {row}")
                    continue

                tokens = tokenizer.encode(prompt.strip() + " " + response.strip())
                actual_length = len(tokens)
                if actual_length > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens = tokens + [tokenizer.pad_token_id] * (max_length - actual_length)

                # Ensure pad_token_id is valid and replace None values
                if tokenizer.pad_token_id is None:
                    raise ValueError("pad_token_id is None. Ensure the tokenizer is configured correctly.")
                tokens = [token if token is not None else tokenizer.pad_token_id for token in tokens]

                input_ids = tokens[:-1]
                labels = tokens[1:]
                yield {'input_ids': input_ids, 'labels': labels}
            gc.collect()  # Explicitly call garbage collector to free up memory

            log_memory_usage()  # Log memory usage at the end of the chunk

    gc.collect()  # Explicitly call garbage collector at the start

# Initialize model only once
def model_fn(inputs, config):
    model = VishwamAILLM(config=config)
    return model

def main():
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Current working directory: {os.getcwd()}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reduce model complexity
    config['embed_dim'] = 128  # Further reduce embedding dimension
    config['num_heads'] = 4  # Further reduce number of attention heads
    config['num_layers'] = 2  # Further reduce number of layers

    # Initialize memory usage log file
    memory_log_file = '/home/ubuntu/chat-agent/memory_usage.txt'
    with open(memory_log_file, 'w') as f:
        f.write("Timestamp,Memory_Usage(MiB)\n")

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

    # Initialize model
    model = model_fn(None, config)
    from flax.training import checkpoints
    model_state = checkpoints.restore_checkpoint(ckpt_dir=config['model_name'], target=model)
    model_params = model_state['params']

    # Create datasets with smaller subsets of data for incremental training
    train_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    eval_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_dialogues.csv'))
    train_dataset = create_dataset_from_csv(train_file_path, tokenizer, 1, config['max_seq_length'])
    eval_dataset = create_dataset_from_csv(eval_file_path, tokenizer, 1, config['max_seq_length'])

# Initialize model only once
def model_fn(inputs, config):
    model = VishwamAILLM(config=config)
    return model

from bias_analysis import analyze_bias

def update(params, opt_state, batch):
    def loss_fn_wrapper(params):
        logits, _ = model.apply(params, batch['input_ids'])
        return loss_fn(logits, batch['labels'])

    grads = jax.grad(loss_fn_wrapper)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    x1, x2 = jnp.split(x, 2, axis=-1)
    sin = sin.reshape(x1.shape)
    cos = cos.reshape(x1.shape)
    x_rotated = (x1 * cos) + (rotate_half(x1) * sin)
    return jnp.concatenate([x_rotated, x2], axis=-1)

class VishwamAILLM(nn.Module):
    config: Dict

    def setup(self):
        config_with_head_dim = {**self.config, 'head_dim': 32}  # Add head_dim to the configuration
        self.transformer = ImprovedVishwamAIModel(config_with_head_dim)
        self.lm_head = nn.Dense(self.config['vocab_size'])

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        transformer_outputs, new_kv_cache = self.transformer(inputs, is_training, kv_cache)
        lm_logits = self.lm_head(transformer_outputs)
        return lm_logits, new_kv_cache

from generate_modular_question import generate_modular_question
from typing import Iterable
import more_itertools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add file handler to write logs to a file
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/train.log'))
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import VishwamAILLM
from src.training.trainer import VishwamAITrainer
from src.environment.chat_agent_env import ChatAgentEnv
from stable_baselines3 import PPO

def create_dataset_from_csv(file_path: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    def load_and_preprocess_data(file_path: str):
        chunk_size = 25  # Further reduce chunk size to manage memory usage
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            logger.info(f"Loaded data chunk from CSV: {chunk.head()}")
            for _, row in chunk.iterrows():
                prompt = row['prompt']
                response = row['response']

                # Check for empty or None values in prompt and response
                if not prompt or not response:
                    logger.warning(f"Warning: Empty or None value encountered in row: {row}")
                    continue

                tokens = tokenizer.encode(prompt.strip() + " " + response.strip())
                actual_length = len(tokens)
                if actual_length > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens = tokens + [tokenizer.pad_token_id] * (max_length - actual_length)

                # Ensure pad_token_id is valid and replace None values
                if tokenizer.pad_token_id is None:
                    raise ValueError("pad_token_id is None. Ensure the tokenizer is configured correctly.")
                tokens = [token if token is not None else tokenizer.pad_token_id for token in tokens]

                input_ids = tokens[:-1]
                labels = tokens[1:]
                yield {'input_ids': input_ids, 'labels': labels}
            gc.collect()  # Explicitly call garbage collector to free up memory

    def create_batch(samples):
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

        batch = {
            'input_ids': jnp.array(input_ids),
            'labels': jnp.array(labels)
        }
        return batch

    dataset = load_and_preprocess_data(file_path)
    batched_dataset = (create_batch(samples) for samples in more_itertools.chunked(dataset, batch_size))
    return batched_dataset

def update_dataset_with_new_data(existing_dataset: Iterable, new_data_file: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    new_data = create_dataset_from_csv(new_data_file, tokenizer, batch_size, max_length)
    updated_dataset = more_itertools.chain(existing_dataset, new_data)
    return updated_dataset

import gc


if __name__ == "__main__":
    gc.collect()  # Explicitly call garbage collector at the start
    main()
