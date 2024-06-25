import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_text as tf_text
import sys
import os

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

def main():
    # Sample input prompt
    prompt = "Once upon a time"

    # Read the SentencePiece model file
    model_path = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm"
    with tf.io.gfile.GFile(model_path, "rb") as f:
        model_proto = f.read()

    # Initialize the SentencepieceTokenizer with the model proto
    try:
        tokenizer = tf_text.SentencepieceTokenizer(
            model=model_proto,
            out_type=tf.int32,
            nbest_size=-1,
            alpha=1.0,
            add_bos=False,
            add_eos=False,
            reverse=False
        )
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

    # Tokenize the input prompt
    try:
        tokenized_prompt = tokenizer.tokenize([prompt]).numpy()
        print(f"Tokenized prompt: {tokenized_prompt}")
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        return

    # Transform the forward function
    transformed_model = hk.transform(forward_fn)

    # Generate text
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array(tokenized_prompt, dtype=jnp.int32))
    generated_text = transformed_model.apply(params, rng, jnp.array(tokenized_prompt, dtype=jnp.int32))
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
