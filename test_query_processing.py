import unittest
import query_processing
from query_processing import process_query
from content_generation import generate_content, is_content_relevant

class TestQueryProcessing(unittest.TestCase):

    def test_valid_query(self):
        query = "What is the impact of generative AI on search engines?"
        result = process_query(query)
        self.assertIn("tokens", result)
        self.assertIn("pos_tags", result)
        self.assertIn("named_entities", result)
        self.assertIsInstance(result["tokens"], list)
        self.assertIsInstance(result["pos_tags"], list)
        self.assertIsInstance(result["named_entities"], list)
        self.assertEqual(result["tokens"], ["What", "is", "the", "impact", "of", "generative", "AI", "on", "search", "engines", "?"])
        self.assertEqual(result["pos_tags"], ["PRON", "AUX", "DET", "NOUN", "ADP", "ADJ", "PROPN", "ADP", "NOUN", "NOUN", "PUNCT"])
        self.assertEqual(result["named_entities"], [("AI", "ORG")])

    def test_empty_query(self):
        query = ""
        result = process_query(query)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid query. Please provide a non-empty string.")

    def test_invalid_query_type(self):
        query = 12345
        result = process_query(query)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid query. Please provide a non-empty string.")

    def test_spacy_model_not_loaded(self):
        original_nlp = query_processing.nlp
        try:
            query_processing.nlp = None
            query = "What is the impact of generative AI on search engines?"
            result = process_query(query)
            self.assertIn("error", result)
            self.assertEqual(result["error"], "spaCy model not loaded")
        finally:
            query_processing.nlp = original_nlp

    def test_is_content_relevant(self):
        query_tokens = ["What", "is", "the", "impact", "of", "generative", "AI", "on", "search", "engines", "?"]
        generated_text_relevant = "The impact of generative AI on search engines is significant."
        generated_text_not_relevant = "This text is not related to the query at all."

        self.assertTrue(is_content_relevant(query_tokens, generated_text_relevant))
        self.assertFalse(is_content_relevant(query_tokens, generated_text_not_relevant))

    def test_generate_content(self):
        processed_query = {
            "tokens": ["What", "is", "the", "impact", "of", "generative", "AI", "on", "search", "engines", "?"],
            "pos_tags": ["PRON", "AUX", "DET", "NOUN", "ADP", "ADJ", "PROPN", "ADP", "NOUN", "NOUN", "PUNCT"],
            "named_entities": [("AI", "ORG")]
        }
        result = generate_content(processed_query, max_length=50)
        self.assertIn("generated_content", result)
        self.assertIsInstance(result["generated_content"], str)
        self.assertGreater(len(result["generated_content"]), 20)
        self.assertNotIn("error", result["generated_content"].lower())
        self.assertTrue(is_content_relevant(processed_query["tokens"], result["generated_content"]))

    def test_generate_content_invalid_max_length(self):
        processed_query = {
            "tokens": ["What", "is", "the", "impact", "of", "generative", "AI", "on", "search", "engines", "?"],
            "pos_tags": ["PRON", "AUX", "DET", "NOUN", "ADP", "ADJ", "PROPN", "ADP", "NOUN", "NOUN", "PUNCT"],
            "named_entities": [("AI", "ORG")]
        }
        result = generate_content(processed_query, max_length=1000)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid max_length parameter. It must be a positive integer and less than or equal to 500.")

if __name__ == "__main__":
    unittest.main()
