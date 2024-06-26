import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece as spm
import sys
import os
import time

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_architecture import VishwamAIModel

def forward_fn(input_ids):
    # Model parameters
    vocab_size = 20000
    embed_dim = 512
    num_heads = 8
    num_layers = 12
    num_experts = 4
    max_sequence_length = 1024

    # Instantiate the model within the hk.transform context
    model = VishwamAIModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        max_sequence_length=max_sequence_length
    )

    return model(input_ids)

def generate_text(model, params, rng, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.tokenize([prompt]).numpy()
    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    generated_ids = input_ids

    start_time = time.time()

    for _ in range(max_length):
        logits = model.apply(params, rng, generated_ids)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=-1)
        print(f"Generated token ID: {next_token}")
        if next_token == tokenizer.token_to_id("[EOS]"):
            break

    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Token generation time: {generation_time:.2f} seconds")

    return tokenizer.detokenize(generated_ids)[0]

def main():
    # Sample input prompt
    prompt = "Once upon a time"

    # Read the serialized SentencePiece model file
    model_path = "/home/ubuntu/chat-agent/VishwamAI/scripts/vishwamai.serialized"
    try:
        print("Loading SentencePiece model...")
        start_time = time.time()
        with open(model_path, 'rb') as f:
            model_proto = f.read()
        end_time = time.time()
        print(f"Model proto type: {type(model_proto)}")
        print(f"Model proto length: {len(model_proto)}")
        print(f"Model proto content: {model_proto[:100]}")  # Print first 100 bytes for inspection
        print(f"Loading SentencePiece model took {end_time - start_time:.2f} seconds")

        print("Initializing tokenizer...")
        start_time = time.time()
        tokenizer = tf_text.SentencepieceTokenizer(
            model=model_proto,
            out_type=tf.int32,
            nbest_size=0,  # Set nbest_size to 0 to disable n-best sampling
            alpha=1.0,
            add_bos=False,
            add_eos=False,
            reverse=False
        )
        end_time = time.time()
        print("Tokenizer initialized successfully.")
        print(f"Tokenizer initialization took {end_time - start_time:.2f} seconds")
    except tf.errors.InvalidArgumentError as e:
        print(f"TensorFlow InvalidArgumentError initializing tokenizer: {e}")
        return
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

    # Transform the forward function
    print("Transforming the forward function...")
    start_time = time.time()
    transformed_model = hk.transform(forward_fn)
    end_time = time.time()
    print(f"Transforming the forward function took {end_time - start_time:.2f} seconds")

    # Generate text
    print("Generating text...")
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Initialize with dummy input
    generated_text = generate_text(transformed_model, params, rng, tokenizer, prompt)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
