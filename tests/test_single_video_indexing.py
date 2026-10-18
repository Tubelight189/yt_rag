import unittest
from unittest.mock import Mock

from src.youtube.videos import parse_video_id


class SingleVideoTests(unittest.TestCase):
    def test_parses_supported_urls(self):
        expected = "dQw4w9WgXcQ"
        for value in (
            expected,
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=4",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        ):
            self.assertEqual(parse_video_id(value), expected)

    def test_rejects_non_youtube_url(self):
        with self.assertRaises(ValueError):
            parse_video_id("https://example.com/video")


if __name__ == "__main__":
    unittest.main()
