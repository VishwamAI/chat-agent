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

def load_data(file_path, max_seq_length=50, batch_size=2):
    """
    Load and preprocess the training data.
    Args:
        file_path: str. Path to the text data file.
        max_seq_length: int. Maximum sequence length for padding/truncation.
        batch_size: int. Batch size for training.
    Returns:
        tf.data.Dataset. Preprocessed dataset.
    """
    dataset = tf.data.TextLineDataset(file_path)
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=VOCAB_FILE, sequence_length=max_seq_length)
    dataset = dataset.map(lambda x: tokenizer.tokenize(x))
    dataset = dataset.padded_batch(batch_size, padded_shapes=[max_seq_length])
    return dataset

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

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, optimizer.init(params))
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

@profile
def train_model(data_file, num_epochs=10):
    """
    Train the VishwamAI model.
    Args:
        data_file: str. Path to the text data file.
        num_epochs: int. Number of training epochs.
    """
    # Load and preprocess the data
    dataset = load_data(data_file, batch_size=2)  # Further reduced batch size for memory optimization

    # Initialize the model and optimizer
    model = hk.transform(lambda x: VishwamAIModel()(x))
    optimizer = optax.adam(learning_rate=1e-3)
    rng = jax.random.PRNGKey(42)

    # Initialize model parameters
    example_batch = next(iter(dataset))
    example_batch = example_batch.numpy().tolist()  # Convert tensor to list of lists of integers
    example_batch = jax.numpy.array(example_batch, dtype=jnp.int32)  # Convert to int32
    params = model.init(rng, example_batch)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataset:
            batch = batch.numpy().tolist()  # Convert tensor to list of lists of integers
            batch = jax.numpy.array(batch, dtype=jnp.int32)  # Convert to int32
            logging.info(f"Data type of batch before model apply: {batch.dtype}")
            loss, params, opt_state = train_step(params, model, optimizer, batch, rng)
            logging.info(f"Epoch {epoch + 1}, Loss: {loss}")

    # Save the trained model
    with open("vishwamai_model_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logging.info("Model training complete and parameters saved.")

if __name__ == "__main__":
    data_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/text_data.txt"
    train_model(data_file)
