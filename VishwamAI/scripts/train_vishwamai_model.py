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

def data_generator(file_path, max_seq_length=32, batch_size=8, label_encoder=None):
    """
    Generator function to yield batches of data and corresponding labels.
    Args:
        file_path: str. Path to the text data file.
        max_seq_length: int. Maximum sequence length for padding/truncation.
        batch_size: int. Number of samples per batch.
        label_encoder: dict. A dictionary mapping string labels to integer indices.
    Yields:
        tuple: A batch of tokenized and padded data and corresponding labels.
    """
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=VOCAB_FILE, sequence_length=max_seq_length)

    with open(file_path, 'r') as f:
        batch_lines = []
        batch_labels = []
        for line in f:
            input_data, label = line.strip().split('\t')  # Assuming tab-separated input and label
            batch_lines.append(input_data)
            batch_labels.append(label_encoder[label] if label_encoder else label)
            if len(batch_lines) == batch_size:
                tokenized_batch = [tokenizer.tokenize(line) for line in batch_lines]
                padded_batch = [tf.pad(tokens, [[0, max_seq_length - tf.shape(tokens)[0]]], constant_values=0) for tokens in tokenized_batch]
                yield tf.convert_to_tensor(padded_batch, dtype=tf.int32), tf.convert_to_tensor(batch_labels, dtype=tf.int32)
                batch_lines = []
                batch_labels = []
        if batch_lines:
            tokenized_batch = [tokenizer.tokenize(line) for line in batch_lines]
            padded_batch = [tf.pad(tokens, [[0, max_seq_length - tf.shape(tokens)[0]]], constant_values=0) for tokens in tokenized_batch]
            yield tf.convert_to_tensor(padded_batch, dtype=tf.int32), tf.convert_to_tensor(batch_labels, dtype=tf.int32)

def train_step(params, model, optimizer, batch, labels, rng):
    """
    Perform a single training step.
    Args:
        params: hk.Params. Model parameters.
        model: VishwamAIModel. The model to be trained.
        optimizer: optax.GradientTransformation. The optimizer for training.
        batch: tf.Tensor. A batch of input data.
        labels: tf.Tensor. The target labels corresponding to the input data.
        rng: jax.random.PRNGKey. Random number generator key.
    Returns:
        loss: jnp.ndarray. The loss value for the batch.
        new_params: hk.Params. Updated model parameters.
        new_opt_state: optax.OptState. Updated optimizer state.
    """
    def loss_fn(params):
        logits = model.apply(params, rng, batch)  # logits shape: [batch_size, num_classes]
        one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])  # labels shape: [batch_size]
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
        return loss

    # Use mixed precision training
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree_map(lambda g: g.astype(jnp.float32), grads)  # Cast gradients back to float32
    updates, new_opt_state = optimizer.update(grads, optimizer.init(params))
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

@profile  # Uncommenting the memory profiling decorator to identify memory usage spikes
def train_model(data_file, num_epochs=10, batch_size=8):
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

    # Initialize label encoder
    label_encoder = {
        "complaint": 0,
        "inquiry": 1,
        "praise": 2
    }

    # Initialize model parameters
    example_batch, example_labels = next(data_generator(data_file, batch_size=batch_size, label_encoder=label_encoder))
    example_batch = example_batch.numpy().tolist()  # Convert tensor to list of lists of integers
    example_batch = jax.numpy.array(example_batch, dtype=jnp.int32)  # Convert to int32
    example_labels = example_labels.numpy().tolist()  # Convert tensor to list of labels
    example_labels = jax.numpy.array(example_labels, dtype=jnp.int32)  # Convert to int32
    params = model.init(rng, example_batch)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_generator(data_file, batch_size=batch_size, label_encoder=label_encoder):
            batch, labels = batch
            batch = batch.numpy().tolist()  # Convert tensor to list of lists of integers
            batch = jax.numpy.array(batch, dtype=jnp.int32)  # Convert to int32
            labels = labels.numpy().tolist()  # Convert tensor to list of labels
            labels = jax.numpy.array(labels, dtype=jnp.int32)  # Convert to int32
            logging.info(f"Data type of batch before model apply: {batch.dtype}")
            loss, params, opt_state = train_step(params, model, optimizer, batch, labels, rng)
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
