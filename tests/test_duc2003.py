import os
import sys
import shutil
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase

class TestDUC2003(DatasetTestCase):

    def test_download(self):
        r = chazutsu.datasets.DUC2003().download(self.test_dir)
        self.assertTrue(len(r.data().columns), 5)  # news and 4summary
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)
        print(r.train_data().head(5))
        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
