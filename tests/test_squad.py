import os
import sys
import shutil
import unittest
import chazutsu.datasets

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data/squad_train")
if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)


class TestSquad(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)

    def test_download(self):
        r_train = chazutsu.datasets.Squad("train").download(DATA_ROOT)
        r_dev = chazutsu.datasets.Squad("dev").download(DATA_ROOT)

        train = r_train.data()
        dev = r_dev.data()

        self.assertEqual(train.shape[0], 87907)
        self.assertEqual(dev.shape[0], 36497)
        self.assertEqual(train.shape[1], dev.shape[1])

        shutil.rmtree(r_train.root)
        shutil.rmtree(r_dev.root)


if __name__ == "__main__":
    unittest.main()
