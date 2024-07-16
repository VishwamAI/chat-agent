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
from huggingface_hub import HfApi

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
    mmlu = load_dataset("cais/mmlu", "mathematics")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    def preprocess_function(examples):
        inputs = [f"Question: {q}\nChoices: A) {a} B) {b} C) {c} D) {d}\nAnswer:" for q, a, b, c, d in zip(examples['question'], examples['choices'][0], examples['choices'][1], examples['choices'][2], examples['choices'][3])]
        targets = [f" {examples['answer'][i]}" for i in range(len(examples['answer']))]
        tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        tokenized_targets = tokenizer(targets, truncation=True, padding="max_length", max_length=8)

        tokenized_inputs["labels"] = tokenized_targets["input_ids"]
        return tokenized_inputs

    processed_dataset = mmlu["train"].map(preprocess_function, batched=True, remove_columns=mmlu["train"].column_names)
    return processed_dataset

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
        # MMLU-specific configuration parameters
        num_choices=4,
        max_seq_length=512,
    )

def create_train_state(rng, config):
    model = NextGenJAXModel(config)
    params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))
    tx = optax.adamw(learning_rate=1e-5)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['labels'][:, 1:]
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state, eval_ds):
    correct = 0
    total = 0
    for batch in eval_ds:
        logits = state.apply_fn(state.params, batch['input_ids'])
        predictions = jnp.argmax(logits[:, -1, :], axis=-1)
        correct += jnp.sum(predictions == batch['labels'][:, -1])
        total += len(predictions)
    accuracy = correct / total
    return accuracy

def save_checkpoint(state, path):
    with open(path, 'wb') as f:
        f.write(jax.device_get(jax.tree_util.tree_map(lambda x: x.copy(), state)))

def train_vishwamai():
    rng = jax.random.PRNGKey(0)
    config = create_vishwamai_config()
    state = create_train_state(rng, config)

    dataset = load_datasets()
    eval_dataset = load_dataset("cais/mmlu", "mathematics", split="validation")

    for epoch in range(10):  # Adjust number of epochs as needed
        for batch in dataset:
            state, loss = train_step(state, batch)
            print(f"Epoch {epoch}, Loss: {loss}")

        mmlu_score = evaluate(state, eval_dataset)
        print(f"Epoch {epoch}, MMLU Math Score: {mmlu_score}")
        save_checkpoint(state, f"checkpoint_epoch_{epoch}")

    return state

def upload_model_to_hub(state, model_name, version):
    # Convert state to Hugging Face model format
    config = create_vishwamai_config()
    model = NextGenJAXModel(config)
    model.params = state.params

    # Save model locally
    model.save_pretrained(f"{model_name}-{version}")

    # Upload to Hugging Face Hub
    api = HfApi()
    api.upload_folder(
        folder_path=f"{model_name}-{version}",
        repo_id=f"your-username/{model_name}",
        repo_type="model",
    )

if __name__ == "__main__":
    final_state = train_vishwamai()
    upload_model_to_hub(final_state, "vishwamai-mmlu-math", "v1.0")