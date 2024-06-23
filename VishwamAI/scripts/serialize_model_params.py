import haiku as hk
import jax
import numpy as np
from model_architecture import VishwamAIModel

def serialize_model_params():
    model = hk.transform(lambda x: VishwamAIModel()(x))
    rng = jax.random.PRNGKey(42)
    dummy_input = jax.numpy.ones((1, 1024), dtype=jax.numpy.int32)
    params = model.init(rng, dummy_input)
    params_dict = hk.data_structures.to_immutable_dict(params)
    with open('vishwamai_model_params.pkl', 'wb') as f:
        np.savez(f, **params_dict)

if __name__ == "__main__":
    serialize_model_params()
