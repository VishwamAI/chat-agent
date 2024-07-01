import unittest
from src.features.audio_to_text import audio_to_text

class TestAudioToText(unittest.TestCase):
    def test_audio_to_text_valid(self):
        # Test with a valid audio file
        result = audio_to_text("tests/test_files/sample_audio.wav")
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)

    def test_audio_to_text_invalid(self):
        # Test with an invalid audio file
        result = audio_to_text("tests/test_files/invalid_audio.wav")
        self.assertIn("Error:", result)

if __name__ == "__main__":
    unittest.main()
