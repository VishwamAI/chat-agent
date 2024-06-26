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

    # Pass the raw dummy input to the model
    params = model.init(rng, dummy_input)
    params_dict = hk.data_structures.to_immutable_dict(params)
    with open('vishwamai_model_params.pkl', 'wb') as f:
        np.savez(f, **params_dict)

if __name__ == "__main__":
    serialize_model_params()
