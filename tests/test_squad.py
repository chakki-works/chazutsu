import os
import sys
import shutil
import unittest
import chazutsu.datasets

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data/squad_train")
if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)


class TestSquad(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)

    def test_download_v1_1(self):
        r_train = chazutsu.datasets.SQuAD(
            kind="train", version="v1.1").download(DATA_ROOT)
        r_dev = chazutsu.datasets.SQuAD(
            kind="dev", version="v1.1").download(DATA_ROOT)

        train = r_train.data()
        dev = r_dev.data()

        self.assertEqual(train.shape[0], 87599)
        self.assertEqual(dev.shape[0], 10570)
        self.assertEqual(train.shape[1], dev.shape[1])

        shutil.rmtree(r_train.root)
        shutil.rmtree(r_dev.root)

    def test_download_v2_0(self):
        r_train = chazutsu.datasets.SQuAD(
            kind="train", version="v2.0").download(DATA_ROOT)
        r_dev = chazutsu.datasets.SQuAD(
            kind="dev", version="v2.0").download(DATA_ROOT)

        train = r_train.data()
        dev = r_dev.data()

        self.assertEqual(train.shape[0], 130319)
        self.assertEqual(dev.shape[0], 11873)
        self.assertEqual(train.shape[1], dev.shape[1])

        shutil.rmtree(r_train.root)
        shutil.rmtree(r_dev.root)


if __name__ == "__main__":
    unittest.main()
