import tensorflow as tf
import keras_nlp
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import logging
import pickle
from model_architecture import VishwamAIModel
from config import VOCAB_FILE
from memory_profiler import profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_generator(file_path, max_seq_length=32, batch_size=16):
    """
    Generator function to yield batches of data.
    Args:
        file_path: str. Path to the text data file.
        max_seq_length: int. Maximum sequence length for padding/truncation.
        batch_size: int. Number of samples per batch.
    Yields:
        tf.Tensor. A batch of tokenized and padded data.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=VOCAB_FILE, sequence_length=max_seq_length)

    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        tokenized_batch = [tokenizer.tokenize(line) for line in batch_lines]
        padded_batch = [tf.pad(tokens, [[0, max_seq_length - tf.shape(tokens)[0]]], constant_values=0) for tokens in tokenized_batch]
        yield tf.convert_to_tensor(padded_batch, dtype=tf.int32)

def train_step(params, model, optimizer, batch, rng):
    """
    Perform a single training step.
    Args:
        params: hk.Params. Model parameters.
        model: VishwamAIModel. The model to be trained.
        optimizer: optax.GradientTransformation. The optimizer for training.
        batch: tf.Tensor. A batch of input data.
        rng: jax.random.PRNGKey. Random number generator key.
    Returns:
        loss: jnp.ndarray. The loss value for the batch.
        new_params: hk.Params. Updated model parameters.
        new_opt_state: optax.OptState. Updated optimizer state.
    """
    def loss_fn(params):
        logits = model.apply(params, rng, batch)
        labels = jax.nn.one_hot(batch, num_classes=logits.shape[-1])
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        return loss

    # Apply gradient checkpointing
    loss_fn = jax.checkpoint(loss_fn)

    # Use mixed precision training
    loss, grads = jax.value_and_grad(loss_fn, dtype=jnp.float16)(params)
    grads = jax.tree_map(lambda g: g.astype(jnp.float32), grads)  # Cast gradients back to float32
    updates, new_opt_state = optimizer.update(grads, optimizer.init(params))
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

# @profile  # Commenting out the memory profiling decorator to reduce memory overhead
def train_model(data_file, num_epochs=10, batch_size=16):
    """
    Train the VishwamAI model.
    Args:
        data_file: str. Path to the text data file.
        num_epochs: int. Number of training epochs.
        batch_size: int. Number of samples per batch.
    """
    # Initialize the model and optimizer
    model = hk.transform(lambda x: VishwamAIModel()(x))
    optimizer = optax.adam(learning_rate=1e-3)
    rng = jax.random.PRNGKey(42)

    # Initialize model parameters
    example_batch = next(data_generator(data_file, batch_size=batch_size))
    example_batch = example_batch.numpy().tolist()  # Convert tensor to list of lists of integers
    example_batch = jax.numpy.array(example_batch, dtype=jnp.int32)  # Convert to int32
    params = model.init(rng, example_batch)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_generator(data_file, batch_size=batch_size):
            batch = batch.numpy().tolist()  # Convert tensor to list of lists of integers
            batch = jax.numpy.array(batch, dtype=jnp.int32)  # Convert to int32
            logging.info(f"Data type of batch before model apply: {batch.dtype}")
            loss, params, opt_state = train_step(params, model, optimizer, batch, rng)
            logging.info(f"Epoch {epoch + 1}, Loss: {loss}")

            # Explicit garbage collection
            import gc
            gc.collect()

    # Save the trained model
    with open("vishwamai_model_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logging.info("Model training complete and parameters saved.")

if __name__ == "__main__":
    data_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/text_data.txt"
    train_model(data_file)
