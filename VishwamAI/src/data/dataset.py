import jax.numpy as jnp
from typing import Dict, Iterable
import more_itertools

def create_dataset(file_path: str, tokenizer, batch_size: int, max_length: int) -> Iterable:
    def load_and_preprocess_data(file_path: str):
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        
        for line in data:
            tokens = tokenizer.encode(line.strip())
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
            yield {'input_ids': tokens[:-1], 'labels': tokens[1:]}

    def create_batch(samples):
        batch = {
            'input_ids': jnp.array([s['input_ids'] for s in samples]),
            'labels': jnp.array([s['labels'] for s in samples])
        }
        return batch

    dataset = load_and_preprocess_data(file_path)
    batched_dataset = (create_batch(samples) for samples in more_itertools.chunked(dataset, batch_size))
    return batched_dataset