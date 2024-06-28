import unittest
from src.features.video_to_text import video_to_text

class TestVideoToText(unittest.TestCase):
    def test_video_to_text_valid(self):
        # Test with a valid video file
        result = video_to_text("tests/test_files/sample_video.mp4")
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)

    def test_video_to_text_invalid(self):
        # Test with an invalid video file
        result = video_to_text("tests/test_files/invalid_video.mp4")
        self.assertIn("Error:", result)

if __name__ == "__main__":
    unittest.main()
