import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
from model_architecture import VishwamAIModel
import sentencepiece as spm

def serialize_model_params():
    model = hk.transform(lambda x, rng: VishwamAIModel()(x, rng))
    rng = jax.random.PRNGKey(42)
    dummy_input = "This is a dummy input string for testing."  # Adjusted to a single string for tokenization

    # Tokenize the dummy input
    sp = spm.SentencePieceProcessor(model_file='vishwamai.spm')
    tokenized_input = sp.encode(dummy_input, out_type=int)
    tokenized_input = jnp.array(tokenized_input).reshape(1, -1)

    # Pass the tokenized dummy input to the model
    params = model.init(tokenized_input, rng)
    params_dict = hk.data_structures.to_immutable_dict(params)
    with open('vishwamai_model_params.pkl', 'wb') as f:
        np.savez(f, **params_dict)

if __name__ == "__main__":
    serialize_model_params()
