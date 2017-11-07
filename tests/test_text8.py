import os
import sys
import shutil
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
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

    def test_download(self):
        r = chazutsu.datasets.Text8.ja().download(directory=DATA_ROOT)
        shutil.rmtree(r.root)

    def test_tokenize(self):
        r = chazutsu.datasets.Text8.en().download(directory=DATA_ROOT)
        r.make_vocab(min_word_count=5)

        train_ids = r_id.train_data()
        print(train_ids.head(5))
        print(train_ids["sentence"].map(r_id.ids_to_words).head(5))


if __name__ == "__main__":
    unittest.main()
