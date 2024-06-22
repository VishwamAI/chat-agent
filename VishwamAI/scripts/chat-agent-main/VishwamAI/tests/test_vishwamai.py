import unittest
import sys
import os
import haiku as hk
import jax
import jax.numpy as jnp
import timeout_decorator

# Add the directory containing model_architecture.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

from model_architecture import VishwamAIModel, ScoringSystem

class TestVishwamAIModel(unittest.TestCase):

    def setUp(self):
        self.model = VishwamAIModel()
        self.scoring_system = ScoringSystem()

    @timeout_decorator.timeout(300)  # Set a timeout of 5 minutes (300 seconds)
    def test_attention_mechanism(self):
        # Test the attention mechanism with a sample input
        input_text = "What is the capital of France?"
        input_tensor = self.model.tokenizer(input_text, return_tensors="jax").input_ids
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor type: {type(input_tensor)}")
        output = self.model(input_tensor)
        print(f"Output: {output}")
        self.assertIsNotNone(output, "The model output should not be None")

    def test_memory_augmentation(self):
        # Test the memory augmentation with a sample input
        input_text = "What is the capital of France?"
        input_tensor = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)  # Example numerical tensor
        self.model.memory_augmentation.add_memory(input_tensor)
        memory_output = self.model.memory_augmentation(input_tensor)
        self.assertIsNotNone(memory_output, "The memory output should not be None")

    def test_scoring_system(self):
        # Test the scoring system with different question types
        self.assertEqual(self.scoring_system.update_score(True, "easy"), 1, "Score should be 1 for a correct easy question")
        self.assertEqual(self.scoring_system.update_score(True, "medium"), 3, "Score should be 3 for a correct medium question")
        self.assertEqual(self.scoring_system.update_score(True, "hard"), 6, "Score should be 6 for a correct hard question")
        self.assertEqual(self.scoring_system.update_score(False, "easy"), 5, "Score should be 5 after an incorrect easy question")
        self.assertEqual(self.scoring_system.update_score(False, "medium"), 3, "Score should be 3 after an incorrect medium question")
        self.assertEqual(self.scoring_system.update_score(False, "hard"), 0, "Score should be 0 after an incorrect hard question")

    def test_self_improvement(self):
        # Test the self-improvement mechanism
        initial_score = self.scoring_system.score
        self.model.self_improve()
        self.assertGreaterEqual(self.scoring_system.score, initial_score, "The score should not decrease after self-improvement")

if __name__ == "__main__":
    unittest.main()
