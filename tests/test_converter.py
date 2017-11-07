import unittest
import os
import sys
import shutil
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu
from chazutsu.datasets.framework.converter import DictionalyConverter
from chazutsu.datasets.framework.converter import VocabularyConverter
from chazutsu.datasets.framework.converter import OneHotConverter


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestConverter(unittest.TestCase):

    def test_dictionaly_converter(self):
        df = pd.DataFrame({
            "label": ["apple", "banana", "cherry"],
            "data": [1, 0, 2],
        })
        de = DictionalyConverter(df["label"])

        self.assertEqual(de.flow(df["label"]).tolist(), [0, 1, 2])
        self.assertEqual(de.back(df["data"]).tolist(), 
                         ["banana", "apple", "cherry"])

    def test_onehot_converter(self):
        df = pd.DataFrame({
            "label": ["apple", "banana", "cherry", "banana"],
            "data": [1, 0, 2, 1],
        })
        de = OneHotConverter(df["label"])

        converted = de.flow(df["label"])
        self.assertEqual(converted.shape, (4, 3))
        backed = de.back(converted)
        self.assertEqual(backed.tolist(), df["label"].tolist())

    def test_vocabulary_converter(self):
        # download test dataset
        r = chazutsu.datasets.DUC2003(summary_no=1).download(DATA_ROOT)
        v = r.make_vocab(vocab_size=500)
        ve = VocabularyConverter(v, fixed_len=10)

        flowed = ve.flow(r.train_data()["news"][:3])
        self.assertEqual(flowed.shape, (3, 10, 500))
        backed = ve.back(flowed)
        self.assertEqual(backed.shape, (3, 10))

        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
