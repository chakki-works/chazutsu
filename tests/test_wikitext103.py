import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestWikiText103(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        r = chazutsu.datasets.WikiText103().download(directory=DATA_ROOT)

    @classmethod
    def tearDownClass(cls):
        r = chazutsu.datasets.WikiText103().download(directory=DATA_ROOT)
        shutil.rmtree(r.root)

    def test_extract(self):
        r = chazutsu.datasets.WikiText103().download(directory=DATA_ROOT)
        self.assertTrue(len(r.data().columns), 1)
        print(r.train_file_path)
        print(r.test_file_path)
        print(r.valid_file_path)
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)
        self.assertTrue(r.valid_file_path)

    def test_tokenize(self):
        r = chazutsu.datasets.WikiText103().download(directory=DATA_ROOT)
        r.make_vocab(vocab_size=1000, min_word_freq=5)


if __name__ == "__main__":
    unittest.main()
