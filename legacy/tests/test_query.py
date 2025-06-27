import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from tubeatlas.query import query_transcript_db


class TestQuery(unittest.TestCase):
    def setUp(self):
        # Setup temporary directory and SQLite DB
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create a table and insert a test row
        cursor.execute("CREATE TABLE transcripts (video_id TEXT, value TEXT)")
        cursor.execute(
            "INSERT INTO transcripts (video_id, value) VALUES (?, ?)", ("v1", "foo")
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_query_success(self):
        # Should return a DataFrame with the inserted row
        df = query_transcript_db(self.db_path, "SELECT * FROM transcripts")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df.loc[0, "video_id"], "v1")
        self.assertEqual(df.loc[0, "value"], "foo")

    def test_query_missing_db(self):
        # Querying a nonexistent database should return empty DataFrame
        missing = os.path.join(self.temp_dir, "no.db")
        df = query_transcript_db(missing, "SELECT * FROM transcripts")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
