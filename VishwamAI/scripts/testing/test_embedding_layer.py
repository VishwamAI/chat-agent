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

    def test_embedding_with_different_input_lengths(self):
        input_data = tf.constant([[1, 2], [3, 4, 5, 6]], dtype=tf.int32)
        output = self.embedding_layer(input_data)
        expected_shape = (2, 4, self.embed_dim)  # The second dimension should be the length of the longest input
        self.assertEqual(output.shape, expected_shape)

    def test_embedding_with_full_vocab_range(self):
        input_data = tf.constant([list(range(self.vocab_size))], dtype=tf.int32)
        output = self.embedding_layer(input_data)
        expected_shape = (1, self.vocab_size, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_embedding_with_empty_input(self):
        input_data = tf.constant([[]], dtype=tf.int32)
        output = self.embedding_layer(input_data)
        expected_shape = (1, 0, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_embedding_with_out_of_vocab_indices(self):
        input_data = tf.constant([[self.vocab_size + 1]], dtype=tf.int32)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.embedding_layer(input_data)

if __name__ == '__main__':
    unittest.main()
