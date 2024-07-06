import pandas as pd
import logging
from typing import Iterable
import more_itertools
import jax.numpy as jnp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset(file_path: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
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
