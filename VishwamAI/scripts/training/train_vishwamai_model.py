import sys
import os
import tensorflow as tf
import keras_nlp
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import logging
import pickle

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_architecture import VishwamAIModel
from config import VOCAB_FILE
from memory_profiler import profile
import numpy as np
import tensorflow_text as tf_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='../logs/training_run_log.txt')

def data_generator(file_path, max_seq_length=32, batch_size=4, label_encoder=None):
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
    tokenizer = tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(VOCAB_FILE, "rb").read())

    def parse_line(line):
        parts = tf.strings.split(line, '\t')
        if tf.size(parts) < 2:
            dummy_data = tf.zeros([max_seq_length], dtype=tf.int32)
            dummy_label = tf.constant(-1, dtype=tf.int32)
            return dummy_data, dummy_label
        input_data = parts[0]
        label = parts[1]
        tokenized_data = tokenizer.tokenize(input_data)
        tokenized_data = tokenized_data.merge_dims(-2, -1)  # Flatten the tokenized data
        padded_data = tf.pad(tokenized_data, [[0, max_seq_length - tf.shape(tokenized_data)[0]]], constant_values=0)
        label = label_encoder.lookup(label) if label_encoder else label
        # Reduced logging frequency
        if tf.random.uniform([]) < 0.01:  # Log only 1% of the examples
            tf.print(f"Input data: {input_data}")
            tf.print(f"Tokenized data: {tokenized_data}")
            tf.print(f"Padded data: {padded_data}")
            tf.print(f"Label: {label}")
        return padded_data, label

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: y != -1)
    # Removed caching to reduce memory usage
    # dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

import optax

def loss_fn(params, step_rng, batch_jax, labels_jax, transformed_forward):
    logits = transformed_forward.apply(params, step_rng, batch_jax)  # Pass step_rng and batch_jax to the model call
    assert logits.shape == (batch_jax.shape[0], 3), f"Logits shape mismatch: expected ({batch_jax.shape[0]}, 3), got {logits.shape}"
    one_hot_labels = jax.nn.one_hot(labels_jax, num_classes=logits.shape[-1])  # labels shape: [batch_size, num_classes]
    tf.print(f"Logits: {logits}")
    tf.print(f"One-hot labels: {one_hot_labels}")
    tf.print(f"Logits shape: {logits.shape}, One-hot labels shape: {one_hot_labels.shape}")
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
    return loss

def train_step(params, transformed_forward, optimizer, opt_state, batch, labels, step_rng):
    """
    Perform a single training step.
    Args:
        params: dict. Dictionary containing model parameters.
        transformed_forward: hk.Transformed. The transformed forward function.
        optimizer: kfac_jax.Optimizer. The KFAC-JAX optimizer for training.
        opt_state: kfac_jax.OptState. The optimizer state.
        batch: tf.Tensor. A batch of input data.
        labels: tf.Tensor. The target labels corresponding to the input data.
        step_rng: jax.random.PRNGKey. Random number generator key.
    Returns:
        loss: jnp.ndarray. The loss value for the batch.
        new_params: dict. Updated model parameters.
        new_opt_state: kfac_jax.OptState. Updated optimizer state.
    """
    # Ensure inputs are integer dtype for embedding layer
    batch = tf.cast(batch, tf.int32)
    labels = tf.cast(labels, tf.int32)

    # Convert TensorFlow tensors to NumPy arrays before using JAX's data type specification
    batch_np = batch.numpy()
    labels_np = labels.numpy()
    batch_jax = jax.device_put(batch_np)
    labels_jax = jax.device_put(labels_np)

    # Use gradient checkpointing to save memory during the backward pass
    loss, opt_state, stats = optimizer.step(params, opt_state, step_rng, batch=batch_jax, global_step_int=0)
    new_params = opt_state.params
    return loss, new_params, opt_state

@profile
def train_model(data_file, num_epochs=10, batch_size=4):
    """
    Train the VishwamAI model.
    Args:
        data_file: str. Path to the text data file.
        num_epochs: int. Number of training epochs.
        batch_size: int. Number of samples per batch.
    """
    # Remove TensorFlow mixed-precision policy and optimizer setup
    # Ensure that the optimizer and model parameters are correctly configured for JAX and Haiku
    # Set up the K-FAC optimizer
    from kfac_jax import optimizer as kfac_optimizer
    optimizer = kfac_optimizer.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=0.0,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=True,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        initial_damping=1.0,
        multi_device=False,
    )

    def create_model(batch):
        model = VishwamAIModel()
        if not tf.is_tensor(batch):
            batch = tf.convert_to_tensor(batch, dtype=tf.int32)
        return model(batch)

    transformed_forward = hk.transform(create_model)
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
    params = transformed_forward.init(transformer_rng, example_batch)  # Initialize params with transformer_rng and example_batch

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_generator(data_file, batch_size=batch_size, label_encoder=label_encoder):
            batch, labels = batch
            logging.info(f"Data type of batch before model apply: {batch.dtype}")
            rng, step_rng = jax.random.split(rng)
            loss, params, opt_state = train_step(params, transformed_forward, optimizer, opt_state, batch, labels, step_rng)
            logging.info(f"Epoch {epoch + 1}, Loss: {loss}")

        # Save intermediate checkpoint
        checkpoint_file = f"vishwamai_model_params_epoch_{epoch + 1}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(params, f)
        logging.info(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_file}")

        # Explicit garbage collection
        import gc
        gc.collect()

    # Save the final trained model
    with open("vishwamai_model_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logging.info("Model training complete and parameters saved.")

if __name__ == "__main__":
    data_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/text_data_small.txt"
    train_model(data_file)
