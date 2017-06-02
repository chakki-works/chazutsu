import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestPTB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        r = chazutsu.datasets.PTB().download(directory=DATA_ROOT)

    @classmethod
    def tearDownClass(cls):
        r = chazutsu.datasets.PTB().download(directory=DATA_ROOT)
        shutil.rmtree(r.root)

    def test_extract(self):
        r = chazutsu.datasets.PTB().download(directory=DATA_ROOT)
        self.assertTrue(len(r.data().columns), 1)
        self.assertTrue(r.train_file_path)
        self.assertTrue(r.test_file_path)
        self.assertTrue(r.valid_file_path)

    def test_tokenize(self):
        r = chazutsu.datasets.PTB().download(directory=DATA_ROOT)
        r_id = r.to_indexed().make_vocab(min_word_count=5)

        train_ids = r_id.train_data()
        print(train_ids.head(5))
        print(train_ids["sentence"].map(r_id.ids_to_words).head(5))


if __name__ == "__main__":
    unittest.main()
