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
        shutil.rmtree(r.root)

    def test_tokenize(self):
        r = chazutsu.datasets.Text8.en().download(directory=DatasetTestCase.class_test_dir)
        r_id = r.make_vocab(min_word_freq=5)

        train_ids = r.train_data()
        print(train_ids.head(5))
        print(train_ids["sentence"].map(r_id.ids_to_words).head(5))


if __name__ == "__main__":
    unittest.main()
