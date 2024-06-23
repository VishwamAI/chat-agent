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
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_generator(file_path, max_seq_length=32, batch_size=8, label_encoder=None):
    """
    Generator function to yield batches of data and corresponding labels.
    Args:
        file_path: str. Path to the text data file.
        max_seq_length: int. Maximum sequence length for padding/truncation.
        batch_size: int. Number of samples per batch.
        label_encoder: dict. A dictionary mapping string labels to integer indices.
    Returns:
        tf.data.Dataset: A dataset yielding batches of tokenized and padded data and corresponding labels.
    """
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=VOCAB_FILE, sequence_length=max_seq_length)

    def parse_line(line):
        parts = tf.strings.split(line, '\t')
        if tf.size(parts) < 2:
            dummy_data = tf.zeros([max_seq_length], dtype=tf.int32)
            dummy_label = tf.constant(-1, dtype=tf.int32)
            return dummy_data, dummy_label
        input_data = parts[0]
        label = parts[1]
        tokenized_data = tokenizer.tokenize(input_data)
        padded_data = tf.pad(tokenized_data, [[0, max_seq_length - tf.shape(tokenized_data)[0]]], constant_values=0)
        label = label_encoder.lookup(label) if label_encoder else label
        tf.print(f"Input data: {input_data}")
        tf.print(f"Tokenized data: {tokenized_data}")
        tf.print(f"Padded data: {padded_data}")
        tf.print(f"Label: {label}")
        return padded_data, label

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: y != -1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def train_step(params, transformed_forward, optimizer, batch, labels, step_rng):
    """
    Perform a single training step.
    Args:
        params: dict. Dictionary containing model parameters.
        transformed_forward: hk.Transformed. The transformed forward function.
        optimizer: optax.GradientTransformation. The optimizer for training.
        batch: tf.Tensor. A batch of input data.
        labels: tf.Tensor. The target labels corresponding to the input data.
        step_rng: jax.random.PRNGKey. Random number generator key.
    Returns:
        loss: jnp.ndarray. The loss value for the batch.
        new_params: dict. Updated model parameters.
        new_opt_state: optax.OptState. Updated optimizer state.
    """
    # Ensure inputs are integer dtype for embedding layer
    batch = tf.cast(batch, tf.int32)
    labels = tf.cast(labels, tf.int32)

    # Convert TensorFlow tensors to JAX arrays using JAX's data type specification
    batch_jax = jax.device_put(batch.numpy())
    labels_jax = jax.device_put(labels.numpy())

    def loss_fn(params, step_rng):
        logits = transformed_forward.apply(params, step_rng, batch_jax)  # Pass step_rng and batch_jax to the model call
        assert logits.shape == (batch_jax.shape[0], 3), f"Logits shape mismatch: expected ({batch_jax.shape[0]}, 3), got {logits.shape}"
        one_hot_labels = jax.nn.one_hot(labels_jax, num_classes=logits.shape[-1])  # labels shape: [batch_size, num_classes]
        tf.print(f"Logits: {logits}")
        tf.print(f"One-hot labels: {one_hot_labels}")
        tf.print(f"Logits shape: {logits.shape}, One-hot labels shape: {one_hot_labels.shape}")
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
        return loss

    # Use gradient checkpointing to save memory during the backward pass
    loss, grads = jax.value_and_grad(jax.checkpoint(lambda p: loss_fn(p, step_rng)))(params)
    grads = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)  # Cast gradients back to float32
    updates, new_opt_state = optimizer.update(grads, optimizer.init(params))
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

@profile  # Enabling the memory profiling decorator to identify memory usage spikes
def train_model(data_file, num_epochs=10, batch_size=8):
    """
    Train the VishwamAI model.
    Args:
        data_file: str. Path to the text data file.
        num_epochs: int. Number of training epochs.
        batch_size: int. Number of samples per batch.
    """
    def create_model(batch, rng):
        model = VishwamAIModel()
        if not tf.is_tensor(batch):
            batch = tf.convert_to_tensor(batch, dtype=tf.int32)
        return model(batch, rng)

    transformed_forward = hk.transform(create_model)
    optimizer = optax.adam(learning_rate=1e-3)
    rng = jax.random.PRNGKey(42)

    # Initialize label encoder
    keys = tf.constant(["complaint", "inquiry", "praise"])
    values = tf.constant([0, 1, 2])
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
    label_encoder = tf.lookup.StaticHashTable(initializer, default_value=-1)

    # Initialize model parameters
    example_batch, example_labels = next(iter(data_generator(data_file, batch_size=batch_size, label_encoder=label_encoder)))
    example_batch = tf.convert_to_tensor(example_batch, dtype=tf.int32)
    example_labels = tf.convert_to_tensor(example_labels, dtype=tf.int32)
    transformer_rng, rng = jax.random.split(rng)
    params = transformed_forward.init(transformer_rng, example_batch, rng)  # Pass transformer_rng, example_batch, and rng

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_generator(data_file, batch_size=batch_size, label_encoder=label_encoder):
            batch, labels = batch
            logging.info(f"Data type of batch before model apply: {batch.dtype}")
            rng, step_rng = jax.random.split(rng)
            loss, params, opt_state = train_step(params, transformed_forward, optimizer, batch, labels, step_rng)
            logging.info(f"Epoch {epoch + 1}, Loss: {loss}")

        # Explicit garbage collection
        import gc
        gc.collect()

    # Save the trained model
    with open("vishwamai_model_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logging.info("Model training complete and parameters saved.")

if __name__ == "__main__":
    data_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/text_data_corrected.txt"
    train_model(data_file)
