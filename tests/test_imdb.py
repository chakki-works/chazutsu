import os
import sys
import shutil
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase


class TestIMDB(DatasetTestCase):

    def test_extract(self):
        r = chazutsu.datasets.IMDB().download(directory=self.test_dir)
        self.assertTrue(len(r.data().columns), 3)
        self.assertTrue(len(r.unlabeled_data().columns), 1)


if __name__ == "__main__":
    unittest.main()
