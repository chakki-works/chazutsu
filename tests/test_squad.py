import os
import sys
import shutil
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase


class TestSquad(DatasetTestCase):

    def test_download_v1_1(self):
        r_train = chazutsu.datasets.SQuAD(
            kind="train", version="v1.1").download(self.test_dir)
        r_dev = chazutsu.datasets.SQuAD(
            kind="dev", version="v1.1").download(self.test_dir)

        train = r_train.data()
        dev = r_dev.data()

        self.assertEqual(train.shape[0], 87599)
        self.assertEqual(dev.shape[0], 10570)
        self.assertEqual(train.shape[1], dev.shape[1])

    def test_download_v2_0(self):
        r_train = chazutsu.datasets.SQuAD(
            kind="train", version="v2.0").download(self.test_dir)
        r_dev = chazutsu.datasets.SQuAD(
            kind="dev", version="v2.0").download(self.test_dir)

        train = r_train.data()
        dev = r_dev.data()

        self.assertEqual(train.shape[0], 130319)
        self.assertEqual(dev.shape[0], 11873)
        self.assertEqual(train.shape[1], dev.shape[1])


if __name__ == "__main__":
    unittest.main()
