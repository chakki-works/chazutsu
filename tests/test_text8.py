import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestText8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        r = chazutsu.datasets.Text8().download(directory=DATA_ROOT)

    @classmethod
    def tearDownClass(cls):
        r = chazutsu.datasets.Text8().download(directory=DATA_ROOT)
        shutil.rmtree(r.root)

    def test_extract(self):
        r = chazutsu.datasets.Text8().download(directory=DATA_ROOT, test_size=10)
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)

    def test_tokenize(self):
        r = chazutsu.datasets.Text8().download(directory=DATA_ROOT)
        tokenized, vocab = r.tokenize("test", min_word_count=5)
        self.assertTrue(len(tokenized) > 0)

        rev_vocab = {v:k for k, v in vocab.items()}
        print(tokenized[:10])
        print([rev_vocab[i] for i in tokenized[:10]])


if __name__ == "__main__":
    unittest.main()
