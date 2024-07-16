import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from nextgenjax import NextGenJAXModel, NextGenJAXConfig
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import optax
import json
from typing import List, Dict

def load_openhermes_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load and preprocess data from the OpenHermes dataset.

    Args:
    file_path (str): Path to the train.jsonl file.

    Returns:
    List[Dict[str, str]]: A list of dictionaries containing 'input' and 'output' keys.
    """
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            # Split the text into user input and assistant output
            parts = text.split('<|assistant|>')
            if len(parts) == 2:
                user_input = parts[0].replace('<|user|>', '').replace('<|end|>', '').strip()
                assistant_output = parts[1].replace('<|end|>', '').strip()
                dataset.append({
                    'input': user_input,
                    'output': assistant_output
                })
    return dataset

def load_datasets():
    # Load and preprocess datasets
    gemma = load_dataset("google/gemma-2b")
    phi = load_dataset("microsoft/phi-1_5")
    mmlu = load_dataset("cais/mmlu", "mathematics")

    # Load OpenHermes dataset
    openhermes_data = load_openhermes_dataset('/home/ubuntu/chat-agent/openhermes-2.5-phi-3-sft/train.jsonl')
    openhermes_dataset = Dataset.from_dict({
        'text': [f"{item['input']} {item['output']}" for item in openhermes_data]
    })

    # Preprocess and combine datasets
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    combined_dataset = (
        gemma["train"].select(range(1000))
        .concatenate(phi["train"].select(range(1000)))
        .concatenate(mmlu["train"].select(range(1000)))
        .concatenate(openhermes_dataset.select(range(1000)))
    )
    combined_dataset = combined_dataset.map(preprocess_function, batched=True)
    return combined_dataset

def create_vishwamai_config():
    return NextGenJAXConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        intermediate_size=8192,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
    )

def create_train_state(rng, config):
    model = NextGenJAXModel(config)
    params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))
    tx = optax.adamw(learning_rate=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['input_ids'][:, 1:]
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state, eval_ds):
    # TODO: Implement evaluation function
    pass

def train_vishwamai():
    rng = jax.random.PRNGKey(0)
    config = create_vishwamai_config()
    state = create_train_state(rng, config)

    dataset = load_datasets()

    for epoch in range(10):  # Adjust number of epochs as needed
        for batch in dataset:
            state, loss = train_step(state, batch)
            print(f"Epoch {epoch}, Loss: {loss}")

        # TODO: Add evaluation and checkpointing logic

if __name__ == "__main__":
    train_vishwamai()