import os
import sys
import shutil
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase


class TestWikiText103(DatasetTestCase):

    @classmethod
    def setUpClass(cls):
        DatasetTestCase.setUpClass()
        chazutsu.datasets.WikiText103().download(directory=DatasetTestCase.class_test_dir)

    @classmethod
    def tearDownClass(cls):
        r = chazutsu.datasets.WikiText103().download(directory=DatasetTestCase.class_test_dir)
        shutil.rmtree(r.root)

    def test_extract(self):
        r = chazutsu.datasets.WikiText103().download(directory=DatasetTestCase.class_test_dir)
        self.assertTrue(len(r.data().columns), 1)
        print(r.train_file_path)
        print(r.test_file_path)
        print(r.valid_file_path)
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)
        self.assertTrue(r.valid_file_path)

    def test_tokenize(self):
        r = chazutsu.datasets.WikiText103().download(directory=DatasetTestCase.class_test_dir)
        r.make_vocab(vocab_size=1000, min_word_freq=5)


if __name__ == "__main__":
    unittest.main()
