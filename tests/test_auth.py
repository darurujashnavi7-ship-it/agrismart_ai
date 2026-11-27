import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import auth


class HistoryDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir=tempfile.TemporaryDirectory()
        self.database_path=Path(self.temp_dir.name)/"users.db"
        self.database_patch=patch.object(auth,"DB_PATH",self.database_path)
        self.database_patch.start()
        auth.init_db()
        auth.register("alice","password123")
        auth.register("bob","password456")

    def tearDown(self):
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def test_user_can_delete_own_history_entry(self):
        auth.add_history("alice","Yield prediction","Rice: 4.2")
        history_id=auth.get_history("alice")[0][0]

        self.assertTrue(auth.delete_history("alice",history_id))
        self.assertEqual(auth.get_history("alice"),[])

    def test_user_cannot_delete_another_users_history_entry(self):
        auth.add_history("alice","Yield prediction","Rice: 4.2")
        history_id=auth.get_history("alice")[0][0]

        self.assertFalse(auth.delete_history("bob",history_id))
        self.assertEqual(len(auth.get_history("alice")),1)


if __name__=="__main__":
    unittest.main()
