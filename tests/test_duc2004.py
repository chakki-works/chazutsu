import os
import sys
import shutil
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestDUC2004(unittest.TestCase):

    def test_download(self):
        r = chazutsu.datasets.DUC2004().download(DATA_ROOT)
        self.assertTrue(len(r.data().columns), 5)  # news and summary
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)
        print(r.train_data().head(5))
        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
