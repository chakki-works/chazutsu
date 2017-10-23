import os
import sys
import shutil
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestIMDB(unittest.TestCase):

    def test_extract(self):
        r = chazutsu.datasets.IMDB().download(directory=DATA_ROOT)
        self.assertTrue(len(r.data().columns), 3)
        self.assertTrue(len(r.unlabeled_data().columns), 1)

        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
