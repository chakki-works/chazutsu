import os
import sys
import shutil
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase


class TestText8(DatasetTestCase):

    @classmethod
    def setUpClass(cls):
        DatasetTestCase.setUpClass()
        r = chazutsu.datasets.Text8().download(directory=DatasetTestCase.class_test_dir)

    def test_extract(self):
        r = chazutsu.datasets.Text8().download(directory=DatasetTestCase.class_test_dir, test_size=10)
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)

    def test_download(self):
        r = chazutsu.datasets.Text8.ja().download(directory=DatasetTestCase.class_test_dir)
        self.assertIsNotNone(r.train_data())
        self.assertIsNotNone(r.test_data())

    def test_tokenize(self):
        r = chazutsu.datasets.Text8.en().download(directory=DatasetTestCase.class_test_dir)
        r_id = r.make_vocab(min_word_freq=5)


if __name__ == "__main__":
    unittest.main()
