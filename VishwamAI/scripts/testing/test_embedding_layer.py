import tensorflow as tf
import unittest

class TestEmbeddingLayer(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10000
        self.embed_dim = 128
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)

    def test_embedding_output_shape(self):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        output = self.embedding_layer(input_data)
        expected_shape = (2, 3, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_embedding_output_dtype(self):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        output = self.embedding_layer(input_data)
        self.assertEqual(output.dtype, tf.float32)

if __name__ == '__main__':
    unittest.main()
