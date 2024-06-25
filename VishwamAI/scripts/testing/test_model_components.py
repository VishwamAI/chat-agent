import unittest
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from model_architecture import VishwamAIModel

class TestVishwamAIModelComponents(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 20000
        self.embed_dim = 512
        self.num_heads = 8
        self.num_layers = 12
        self.num_experts = 4
        self.max_sequence_length = 1024
        self.model = VishwamAIModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            max_sequence_length=self.max_sequence_length
        )

    def test_embedding_layer(self):
        # Create a dummy input tensor
        dummy_input = jnp.ones((1, self.max_sequence_length), dtype=jnp.int32)

        # Initialize the transformer to access the embedding layer
        transformer_fn = self.model._create_transformer()
        rng = jax.random.PRNGKey(42)
        params = transformer_fn.init(rng, dummy_input)

        # Apply the transformer to the dummy input
        transformer_output = transformer_fn.apply(params, rng, dummy_input)

        # Check the shape of the output
        expected_shape = (1, self.max_sequence_length, self.embed_dim)
        self.assertEqual(transformer_output.shape, expected_shape, "Embedding layer output shape mismatch")

if __name__ == '__main__':
    unittest.main()
