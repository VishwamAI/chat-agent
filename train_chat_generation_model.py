import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.common_utils import shard
from transformers import T5Tokenizer
from models import Transformer, TransformerConfig
import optax
import json
import flax.serialization as flax_serialization
import os
from create_custom_dataset import create_custom_dataset
from chat_agent import AutonomousLearner  # Import AutonomousLearner
import threading  # Import threading module

def create_train_state(rng, config, learning_rate):
    model = Transformer(config=config)
    params = model.init(rng, inputs=jnp.ones((1, 512), jnp.int32), train=True)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch, rng):
    def loss_fn(params):
        rngs = {'dropout': rng}
        logits = state.apply_fn({'params': params}, inputs=batch['input_ids'], train=True, rngs=rngs)
        one_hot_labels = jax.nn.one_hot(batch['labels'], num_classes=logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def data_loader(dataset, batch_size, tokenizer):
    batch = []

    # Add specific training example
    specific_example = {
        'input_ids': tokenizer("hi", padding="max_length", truncation=True, max_length=512)['input_ids'],
        'labels': tokenizer("hello how can I assist you today", padding="max_length", truncation=True, max_length=512)['input_ids']
    }
    batch.append(specific_example)

    # The custom dataset already includes the specific training examples
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def pad_batch(batch, batch_size):
    # Pad the batch to the required batch size
    while len(batch) < batch_size:
        batch.append(batch[0])  # Duplicate the first example to pad the batch
    return batch

def train():
    # Load the custom dataset
    custom_dataset = create_custom_dataset()

    # Initialize the Transformer model and configuration
    config = TransformerConfig(
        vocab_size=32128,
        output_vocab_size=32128,
        emb_dim=512,
        num_heads=8,
        num_layers=6,
        qkv_dim=512,
        mlp_dim=2048,
        max_len=512,  # Adjusted max_len to 512
        dropout_rate=0.3,
        attention_dropout_rate=0.3
    )

    # Create training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config, learning_rate=5e-5)

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Training loop
    batch_size = 4  # Adjust batch size as needed
    for epoch in range(10):  # Increased number of epochs to 10
        for batch in data_loader(custom_dataset, batch_size, tokenizer):
            if len(batch) < batch_size:
                batch = pad_batch(batch, batch_size)  # Pad smaller batches to the required batch size
            batch = {k: jnp.array([example[k] for example in batch]) for k in batch[0].keys()}  # Ensure batch values are arrays
            batch['input_ids'] = jnp.array(batch['input_ids'])  # Ensure input_ids is a jnp array
            batch['input_ids'] = batch['input_ids'].reshape(batch_size, 512)  # Reshape input_ids to (batch, 512)
            print(f"Shape of input_ids: {batch['input_ids'].shape}")  # Print shape of input_ids
            batch = shard(batch)
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, step_rng)
            print(f"Epoch {epoch}, Loss: {loss}")

    # Save the model
    model_bytes = flax_serialization.to_bytes(state.params)
    os.makedirs('./vishwam_model', exist_ok=True)
    with open('./vishwam_model/model_params.msgpack', 'wb') as f:
        f.write(model_bytes)
    tokenizer.save_pretrained('./vishwam_model')

    # Initialize AutonomousLearner and start continuous retraining in a separate thread
    learner = AutonomousLearner()
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Robotics",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Big_data",
        "https://en.wikipedia.org/wiki/Internet_of_things",
        "https://en.wikipedia.org/wiki/Cybersecurity"
    ]
    retrain_thread = threading.Thread(target=learner.continuous_retrain, args=(urls,))
    retrain_thread.start()

    # Placeholder for evaluation mechanism
    # Implement a mechanism to evaluate the model's responses and update its parameters accordingly

if __name__ == "__main__":
    train()
