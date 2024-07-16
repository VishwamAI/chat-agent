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
from nextgenjax import NextGenJAXModel, NextGenJAXConfig
from sklearn.model_selection import train_test_split
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
    audio_dirs = {
        'train': '/home/ubuntu/chat-agent/train',
        'dev': '/home/ubuntu/chat-agent/dev',
        'test': '/home/ubuntu/chat-agent/test'
    }
    datasets = {}

    for split, directory in audio_dirs.items():
        audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
        features = [load_audio(file) for file in audio_files]
        datasets[split] = Dataset.from_dict({'features': features})

    return datasets

def load_audio(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose for (time, features) shape

def create_vishwamai_config():
    return NextGenJAXConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        intermediate_size=8192,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        # Audio-specific configuration parameters
        sample_rate=16000,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        max_audio_length=10,  # in seconds
        num_classes=10,  # Adjust based on your audio classification task
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
        # Assuming batch['features'] contains the audio features
        logits = state.apply_fn(params, batch['features'])
        # Adjust the loss function for audio classification or regression
        # This example assumes a classification task with integer labels
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def evaluate(state, eval_ds):
    total_loss = 0
    total_samples = 0
    for batch in eval_ds:
        features = batch['audio_features']
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
    config = create_vishwamai_config()
    state = create_train_state(rng, config)

    train_dataset = load_datasets('train')
    eval_dataset = load_datasets('dev')

    for epoch in range(10):  # Adjust number of epochs as needed
        for audio_file in train_dataset:
            features = load_audio(audio_file)
            state, loss = train_step(state, features)
            print(f"Epoch {epoch}, Loss: {loss}")

        eval_score = evaluate(state, eval_dataset)
        print(f"Epoch {epoch}, Evaluation Score: {eval_score}")
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