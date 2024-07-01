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
import psutil  # Import psutil for memory profiling

class VishwamAILLM(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = hk.Embed(vocab_size=config['vocab_size'], embed_dim=config['embed_dim'])
        self.lstm = hk.LSTM(hidden_size=config['hidden_size'])
        self.output_layer = hk.Linear(output_size=config['vocab_size'])

    def __call__(self, input_ids, hidden_state=None):
        logger.debug(f"Input IDs: {input_ids}")
        embedded = self.embedding(input_ids)
        logger.debug(f"Embedded: {embedded}")
        if hidden_state is None:
            hidden_state = self.lstm.initial_state(batch_size=embedded.shape[0])
        output, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.output_layer(output)
        logger.debug(f"Logits: {logits}")
        return logits, hidden_state

from transformers import AutoTokenizer

def loss_fn(logits, labels):
    logger.debug(f"Logits: {logits}")
    logger.debug(f"Labels: {labels}")
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    logger.debug(f"Loss: {loss}")
    return loss

from bias_analysis import analyze_bias

def update(params, opt_state, batch):
    def loss_fn_wrapper(params):
        logits, _ = model.apply(params, batch['input_ids'])
        return loss_fn(logits, batch['labels'])

    grads = jax.grad(loss_fn_wrapper)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

from generate_modular_question import generate_modular_question
from typing import Iterable
import more_itertools

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add file handler to write logs to a file
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/train.log'))
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
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
            actual_length = len(tokens)
            if (actual_length > max_length):
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [tokenizer.pad_token_id] * (max_length - actual_length)

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

def update_dataset_with_new_data(existing_dataset: Iterable, new_data_file: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    new_data = create_dataset_from_csv(new_data_file, tokenizer, batch_size, max_length)
    updated_dataset = more_itertools.chain(existing_dataset, new_data)
    return updated_dataset

def main():
    # Initialize memory usage log file
    memory_log_file = '/home/ubuntu/chat-agent/memory_usage.txt'
    with open(memory_log_file, 'w') as f:
        f.write("Timestamp,Memory_Usage(MiB)\n")

    def log_memory_usage():
        logger.debug("log_memory_usage function called")
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.used / (1024 * 1024)  # Convert to MiB
        available_memory = memory_info.available / (1024 * 1024)  # Convert to MiB
        timestamp = time.time()
        logger.debug(f"Memory usage: {memory_usage:.2f} MiB, Available memory: {available_memory:.2f} MiB at timestamp: {timestamp}")
        with open(memory_log_file, 'a') as f:
            f.write(f"{timestamp},{memory_usage:.2f},{available_memory:.2f}\n")

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

    # Define a placeholder reward function
    def reward_function(response: str) -> float:
        # Placeholder reward function that assigns a random reward
        # This should be replaced with a more sophisticated evaluation
        return np.random.rand()

    # Simulated interaction phase
    def simulate_interaction(model, tokenizer, prompts: list) -> list:
        responses = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            output = model.generate(input_ids)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
        return responses

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
    logger.debug(f"Optimizer initialized: {optimizer}")
    opt_state = None
    try:
        rng_key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
        model_params = model.init(rng_key, dummy_input)
        logger.debug(f"Model parameters initialized: {model_params}")
        if not isinstance(model_params, dict):
            raise TypeError(f"model_params is not a dictionary, but a {type(model_params)}")
        logger.debug(f"Model parameters type: {type(model_params)}")
        logger.debug(f"Model parameters structure: {model_params}")
        logger.debug(f"Model parameters before optimizer init: {model_params}")
        # Ensure model_params is a dictionary
        if isinstance(model_params, dict):
            opt_state = optimizer.init(model_params)
        else:
            raise TypeError(f"Expected model_params to be a dictionary, but got {type(model_params)}")
        logger.debug(f"Optimizer state initialized: {opt_state}")
    except TypeError as e:
        logger.error(f"TypeError during optimizer state initialization: {e}")
        logger.debug(f"rng_key: {rng_key}")
        logger.debug(f"dummy_input: {dummy_input}")
        if 'model_params' in locals():
            logger.debug(f"model_params: {model_params}")
        raise

    # Initialize trainer
    trainer = VishwamAITrainer(model, config, optimizer, opt_state)

    # Load prompts from the CSV file
    prompts = []
    with open(train_file_path, 'r') as csvfile:
        reader = pd.read_csv(csvfile)
        prompts = reader['prompt'].tolist()

    env = ChatAgentEnv(model, tokenizer, prompts, config['max_seq_length'])

    # Initialize reinforcement learning algorithm
    rl_model = PPO("MlpPolicy", env, verbose=1)

    # Check for existing checkpoints and load the latest one
    checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        params = np.load(checkpoint_path, allow_pickle=True).item()
    else:
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

            logger.debug(f"Logging memory usage at the beginning of epoch {epoch + 1}")
            log_memory_usage()

            for batch in train_dataset:
                logger.debug(f"Logging memory usage before processing batch {train_steps + 1}")
                log_memory_usage()

                batch['input_ids'] = jnp.array(batch['input_ids'])
                logger.debug(f"Input IDs after conversion to JAX array: {batch['input_ids']}")
                batch['input_ids'] = trainer.preprocess_input(batch['input_ids'])
                logger.debug(f"Input IDs after preprocessing: {batch['input_ids']}")
                batch['input_ids'] = trainer.preprocess_math_input(batch['input_ids'])
                logger.debug(f"Input IDs after math preprocessing: {batch['input_ids']}")
                input_shape = batch['input_ids'].shape
                logger.debug(f"Input shape: {input_shape}")
                params, trainer.opt_state, loss, _ = trainer.train_step(params, trainer.opt_state, batch)
                train_loss += loss
                train_steps += 1

                logger.debug(f"Logging memory usage after processing batch {train_steps}")
                log_memory_usage()

                if train_steps % 100 == 0:
                    logger.info(f"Step {train_steps}: Current Train Loss: {loss:.4f}")
                if train_steps % 100 == 0:
                    logger.info(f"Step {train_steps}: Current Train Loss: {loss:.4f}")

                # Save intermediate checkpoint
                if train_steps % 500 == 0:
                    intermediate_checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{train_steps}.npy')
                    logger.debug(f"Saving intermediate checkpoint to {intermediate_checkpoint_path}")
                    logger.debug(f"Checkpoint parameters before saving: {params}")
                    try:
                        if not os.path.exists(checkpoint_dir):
                            logger.error(f"Checkpoint directory {checkpoint_dir} does not exist.")
                        else:
                            logger.debug(f"Checkpoint directory exists: {checkpoint_dir}")
                            logger.debug(f"Parameters to be saved: {params}")
                            logger.debug(f"Parameters before conversion: {params}")
                            params_dict = hk.data_structures.to_immutable_dict(params)  # Convert params to dictionary
                            logger.debug(f"Parameters after conversion to dictionary: {params_dict}")
                            np.save(intermediate_checkpoint_path, params_dict)
                            logger.debug(f"Intermediate checkpoint saved at {intermediate_checkpoint_path}")
                            if os.path.exists(intermediate_checkpoint_path):
                                logger.info(f"Checkpoint file {intermediate_checkpoint_path} created successfully.")
                            else:
                                logger.error(f"Checkpoint file {intermediate_checkpoint_path} was not created.")
                    except Exception as e:
                        logger.error(f"Failed to save intermediate checkpoint at {intermediate_checkpoint_path}: {e}")
                        logger.debug(f"Exception details: {e}")
                    logger.debug(f"Completed attempt to save intermediate checkpoint at step {train_steps}")

            # Reinforcement learning update
            logger.debug(f"Logging memory usage before reinforcement learning update")
            log_memory_usage()
            rl_model.learn(total_timesteps=1000)
            logger.debug(f"Logging memory usage after reinforcement learning update")
            log_memory_usage()

            eval_metrics = trainer.evaluate(params, eval_dataset)
            logger.debug(f"Logging memory usage after evaluation step")
            log_memory_usage()
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']} completed in {epoch_time:.2f} seconds")
            logger.info(f"Train Loss: {train_loss / train_steps:.4f}")
            logger.info(f"Eval Metrics: {eval_metrics}")

            # Save checkpoint after each epoch
            logger.debug(f"Attempting to save checkpoint after epoch {epoch + 1}")
            checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.npy')
            logger.debug(f"Saving checkpoint to {checkpoint_path} after epoch {epoch + 1}")
            logger.debug(f"Checkpoint parameters before saving: {params}")
            try:
                if not os.path.exists(checkpoint_dir):
                    logger.error(f"Checkpoint directory {checkpoint_dir} does not exist.")
                else:
                    logger.debug(f"Checkpoint directory exists: {checkpoint_dir}")
                    logger.debug(f"Parameters to be saved: {params}")
                    params_dict = hk.data_structures.to_immutable_dict(params)  # Convert params to dictionary
                    logger.debug(f"Parameters after conversion to dictionary: {params_dict}")
                    np.save(checkpoint_path, params_dict)
                    logger.debug(f"Checkpoint parameters: {params}")
                    logger.info(f"Checkpoint saved successfully at {checkpoint_path} after epoch {epoch + 1}")
                    if os.path.exists(checkpoint_path):
                        logger.info(f"Checkpoint file {checkpoint_path} created successfully.")
                        # Verify the contents of the saved file
                        loaded_params = np.load(checkpoint_path, allow_pickle=True)
                        logger.debug(f"Loaded parameters type after saving: {type(loaded_params)}")
                        if isinstance(loaded_params, dict):
                            logger.info(f"Parameters are correctly saved as a dictionary at {checkpoint_path}")
                        else:
                            logger.error(f"Parameters are NOT saved as a dictionary at {checkpoint_path}")
                    else:
                        logger.error(f"Checkpoint file {checkpoint_path} was not created.")
            except Exception as e:
                logger.error(f"Failed to save checkpoint at {checkpoint_path} after epoch {epoch + 1}: {e}")
                logger.debug(f"Exception details: {e}")
            logger.debug(f"Completed attempt to save checkpoint after epoch {epoch + 1}")

            if trainer._should_stop_early(eval_metrics):
                logger.info("Early stopping criteria met. Ending training.")
                break

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        interrupted_checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint_interrupted.npy')
        logger.debug(f"Attempting to save checkpoint due to interruption at {interrupted_checkpoint_path}")
        logger.debug(f"Checkpoint parameters before saving: {params}")
        logger.debug(f"Size of parameters: {params.size}")
        logger.debug(f"Type of parameters: {type(params)}")
        logger.debug(f"Shape of parameters: {params.shape}")
        try:
            params_dict = hk.data_structures.to_immutable_dict(params)  # Convert params to dictionary
            np.save(interrupted_checkpoint_path, params_dict)
            logger.debug(f"Checkpoint parameters: {params}")
            logger.info(f"Checkpoint saved at {interrupted_checkpoint_path} due to interruption.")
            if os.path.exists(interrupted_checkpoint_path):
                logger.info(f"Checkpoint file {interrupted_checkpoint_path} created successfully.")
            else:
                logger.error(f"Checkpoint file {interrupted_checkpoint_path} was not created.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint at {interrupted_checkpoint_path} due to interruption: {e}")
            logger.debug(f"Exception details: {e}")

    logger.info("Training process completed.")

    # Save trained parameters
    # Save final checkpoint
    checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint_final.npy')
    logger.debug(f"Attempting to save final checkpoint to {checkpoint_path}")
    logger.debug(f"Checkpoint parameters before saving: {params}")
    logger.debug(f"Parameters before conversion: {params}")
    try:
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory {checkpoint_dir} does not exist.")
        else:
            params_dict = hk.data_structures.to_immutable_dict(params)  # Convert params to dictionary
            logger.debug(f"Parameters after conversion to dictionary: {params_dict}")
            np.save(checkpoint_path, params_dict)
            logger.debug(f"Checkpoint parameters: {params}")
            logger.info(f"Final checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save final checkpoint at {checkpoint_path}: {e}")

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
