import tensorflow as tf
import haiku as hk
import jax
import jax.numpy as jnp
import pickle
from model_architecture import VishwamAIModel
import tensorflow_text as tf_text

def load_model_params(file_path):
    with open(file_path, "rb") as f:
        params = pickle.load(f)
    return params

def create_model(batch):
    model = VishwamAIModel()
    if not tf.is_tensor(batch):
        batch = tf.convert_to_tensor(batch, dtype=tf.int32)
    return model(batch)

def data_generator(file_path, max_seq_length=32, batch_size=4, label_encoder=None):
    tokenizer = tf_text.BertTokenizer(config.VOCAB_FILE, lower_case=True)

    def parse_line(line):
        parts = tf.strings.split(line, '\t')
        if tf.size(parts) < 2:
            dummy_data = tf.zeros([max_seq_length], dtype=tf.int32)
            dummy_label = tf.constant(-1, dtype=tf.int32)
            return dummy_data, dummy_label
        input_data = parts[0]
        label = parts[1]
        tokenized_data = tokenizer.tokenize(input_data)
        tokenized_data = tokenized_data.merge_dims(-2, -1)
        padded_data = tf.pad(tokenized_data, [[0, max_seq_length - tf.shape(tokenized_data)[0]]], constant_values=0)
        label = label_encoder.lookup(label) if label_encoder else label
        if tf.random.uniform([]) < 0.01:
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

def test_inference(data_file, params_file):
    # Load model parameters
    params = load_model_params(params_file)

    # Initialize label encoder
    keys = tf.constant(["complaint", "inquiry", "praise"])
    values = tf.constant([0, 1, 2])
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
    label_encoder = tf.lookup.StaticHashTable(initializer, default_value=-1)

    # Prepare test data
    example_batch, example_labels = next(iter(data_generator(data_file, batch_size=1, label_encoder=label_encoder)))
    example_batch = tf.convert_to_tensor(example_batch, dtype=tf.int32)
    example_labels = tf.convert_to_tensor(example_labels, dtype=tf.int32)

    # Transform the model
    transformed_forward = hk.transform(create_model)
    rng = jax.random.PRNGKey(42)

    # Perform inference
    logits = transformed_forward.apply(params, rng, example_batch)
    predicted_labels = jnp.argmax(logits, axis=-1)

    print(f"Input: {example_batch}")
    print(f"Logits: {logits}")
    print(f"Predicted labels: {predicted_labels}")
    print(f"True labels: {example_labels}")

if __name__ == "__main__":
    data_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/text_data_small.txt"
    params_file = "/home/ubuntu/chat-agent/VishwamAI/scripts/vishwamai_model_params.pkl"
    test_inference(data_file, params_file)
