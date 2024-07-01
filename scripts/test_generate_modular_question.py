import unittest
from generate_modular_question import generate_modular_question

class TestGenerateModularQuestion(unittest.TestCase):

    def test_valid_modules(self):
        modules = [
            {'input_type': None, 'output_type': 'number', 'question': "What is 2 + 2?"},
            {'input_type': 'number', 'output_type': 'number', 'question': "Multiply the result by 3."},
            {'input_type': 'number', 'output_type': 'number', 'question': "Subtract 5 from the result."}
        ]
        expected_question = "What is 2 + 2? Multiply the result by 3. Subtract 5 from the result."
        self.assertEqual(generate_modular_question(modules), expected_question)

    def test_invalid_modules(self):
        modules = [
            {'input_type': None, 'output_type': 'number', 'question': "What is 2 + 2?"},
            {'input_type': 'text', 'output_type': 'number', 'question': "Multiply the result by 3."}
        ]
        with self.assertRaises(ValueError):
            generate_modular_question(modules)

    def test_empty_modules(self):
        modules = []
        expected_question = ""
        self.assertEqual(generate_modular_question(modules), expected_question)

    def test_single_module(self):
        modules = [
            {'input_type': None, 'output_type': 'number', 'question': "What is 2 + 2?"}
        ]
        expected_question = "What is 2 + 2?"
        self.assertEqual(generate_modular_question(modules), expected_question)

if __name__ == "__main__":
    unittest.main()
