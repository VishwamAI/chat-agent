import jax
import jax.numpy as jnp
import json
import librosa
import numpy as np
import optax
import os
from datasets import Dataset
from flax import linen as nn
from flax.training import train_state
from huggingface_hub import HfApi
from nextgenjax.model import create_model
from sklearn.model_selection import train_test_split
from typing import List, Dict



def load_datasets():
    audio_dirs = {
        'train': '/home/ubuntu/chat-agent/train',
        'dev': '/home/ubuntu/chat-agent/dev',
        'test': '/home/ubuntu/chat-agent/test'
    }
    datasets = {}

    for split, directory in audio_dirs.items():
        audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
        features = [load_audio(file) for file in audio_files]

        # Check if label files exist
        label_files = [f.replace('.wav', '.txt') for f in audio_files]
        labels = []
        for label_file in label_files:
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    labels.append(int(f.read().strip()))
            else:
                labels.append(-1)  # Use -1 as a placeholder for missing labels

        datasets[split] = Dataset.from_dict({'features': features, 'labels': labels})

    return datasets

def load_audio(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose for (time, features) shape

# Configuration parameters for the model
NUM_LAYERS = 24
HIDDEN_SIZE = 2048
NUM_HEADS = 32
DROPOUT_RATE = 0.1

# Audio-specific configuration parameters
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_AUDIO_LENGTH = 10  # in seconds
NUM_CLASSES = 10  # Adjust based on your audio classification task

def create_train_state(rng, num_layers, hidden_size, num_heads, dropout_rate):
    model = create_model(num_layers, hidden_size, num_heads, dropout_rate)
    params = model.init(rng, jnp.ones((1, hidden_size), dtype=jnp.float32))
    tx = optax.adamw(learning_rate=1e-5)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['features'])
        if 'labels' in batch and batch['labels'] is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['labels']
            ).mean()
        else:
            loss = jnp.mean(jnp.square(logits))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state, eval_ds):
    total_loss = 0
    total_samples = 0
    for batch in eval_ds:
        features = batch['features']
        labels = batch['labels']
        logits = state.apply_fn(state.params, features)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        total_loss += loss * features.shape[0]
        total_samples += features.shape[0]
    avg_loss = total_loss / total_samples
    return avg_loss

def save_checkpoint(state, path):
    with open(path, 'wb') as f:
        f.write(jax.device_get(jax.tree_util.tree_map(lambda x: x.copy(), state)))

def train_vishwamai():
    rng = jax.random.PRNGKey(0)

    # Define model parameters
    num_layers = 24
    hidden_size = 2048
    num_heads = 32
    dropout_rate = 0.1

    # Create model using create_model function
    model = create_model(num_layers, hidden_size, num_heads, dropout_rate)

    # Initialize model parameters
    params = model.init(rng, jnp.ones((1, HIDDEN_SIZE), dtype=jnp.float32))

    # Create optimizer
    tx = optax.adamw(learning_rate=1e-5)

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    train_dataset = load_datasets()['train']
    eval_dataset = load_datasets()['dev']

    for epoch in range(10):  # Adjust number of epochs as needed
        for batch in train_dataset:
            state, loss = train_step(state, batch)
            print(f"Epoch {epoch}, Loss: {loss}")

        eval_score = evaluate(state, eval_dataset)
        print(f"Epoch {epoch}, Evaluation Score: {eval_score}")
        save_checkpoint(state, f"checkpoint_epoch_{epoch}")

    return state

def upload_model_to_hub(state, model_name, version):
    # Convert state to Hugging Face model format
    model = create_model(num_layers=24, hidden_size=2048, num_heads=32, dropout_rate=0.1)
    model = model.bind(state.params)

    # Save model locally
    model.save_pretrained(f"{model_name}-{version}")

    # Upload to Hugging Face Hub
    api = HfApi()
    api.upload_folder(
        folder_path=f"{model_name}-{version}",
        repo_id=f"VishwamAI/{model_name}",
        repo_type="model",
    )

if __name__ == "__main__":
    final_state = train_vishwamai()
    upload_model_to_hub(final_state, "vishwamai-mmlu-math", "v1.0")