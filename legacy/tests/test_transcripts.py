import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import tempfile
import unittest

from tubeatlas.transcripts import YouTubeTranscriptManager, get_channel_name_from_url


class DummyService:
    """A dummy YouTube Data API service stub for testing."""

    def videoCategories(self):
        return self

    def list(self, part=None, regionCode=None):
        return self

    def execute(self):
        # Return empty items to skip real API calls
        return {"items": []}


class TestTranscripts(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for file storage
        self.temp_dir = tempfile.mkdtemp()
        self.manager = YouTubeTranscriptManager(
            DummyService(), output_dir=self.temp_dir, storage_type="file"
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    def test_parse_duration_standard(self):
        # PT1H2M3S = 1*3600 + 2*60 + 3 = 3723 seconds
        result = self.manager._parse_duration("PT1H2M3S")
        self.assertEqual(result, 3723)

    def test_parse_duration_empty(self):
        # Empty or None duration returns 0
        self.assertEqual(self.manager._parse_duration(""), 0)
        self.assertEqual(self.manager._parse_duration(None), 0)

    def test_parse_duration_invalid(self):
        # Invalid ISO string returns 0
        self.assertEqual(self.manager._parse_duration("invalid"), 0)

    def test_get_channel_name(self):
        # Should correctly extract the channel name after '@'
        self.assertEqual(
            get_channel_name_from_url("https://www.youtube.com/@chan"), "chan"
        )
        self.assertEqual(get_channel_name_from_url("@chan"), "chan")
        # If no '@', returns the full string
        self.assertEqual(get_channel_name_from_url("chan"), "chan")


if __name__ == "__main__":
    unittest.main()
