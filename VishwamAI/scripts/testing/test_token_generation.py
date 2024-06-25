import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece as spm
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

def generate_text(model, params, rng, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.tokenize([prompt]).numpy()
    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    generated_ids = input_ids

    for _ in range(max_length):
        logits = model.apply(params, rng, generated_ids)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=-1)
        if next_token == tokenizer.token_to_id("[EOS]"):
            break

    return tokenizer.detokenize(generated_ids)[0]

def main():
    # Sample input prompt
    prompt = "Once upon a time"

    # Read the SentencePiece model file
    model_path = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm"
    sp = spm.SentencePieceProcessor()
    try:
        sp.Load(model_path)
        model_proto = sp.serialized_model_proto()
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

    # Transform the forward function
    transformed_model = hk.transform(forward_fn)

    # Generate text
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Initialize with dummy input
    generated_text = generate_text(transformed_model, params, rng, tokenizer, prompt)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
