import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
from chazutsu.datasets.framework.resource import Resource


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestResource(unittest.TestCase):
    TEST_FILES = {
        "train": "data_train.txt",
        "test": "data_test.txt",
        "sample": "data_samples.txt",
        "data": "data.txt"
    }

    @classmethod
    def setUpClass(cls):
        for t in cls.TEST_FILES:
            file = cls.TEST_FILES[t]
            if t == "train":
                content = "\t".join(["o", "good sentence."])
            elif t == "test":
                content = "\t".join(["x", "bad sentence."])
            elif t == "sample":
                content = "\t".join(["x", "bad sentence."])
            else:
                content = "\t".join(["-", "normal sentence."])
            
            with open(os.path.join(DATA_ROOT, file), mode="wb") as f:
                f.write(content.encode("utf-8"))

    @classmethod
    def tearDownClass(cls):
        for t in cls.TEST_FILES:
            file = cls.TEST_FILES[t]
            path = os.path.join(DATA_ROOT, file)
            if os.path.exists(path):
                os.remove(path)

    def test_read_resource(self):
        r = Resource(DATA_ROOT)
        for t in self.TEST_FILES:
            file = self.TEST_FILES[t]
            path = os.path.join(DATA_ROOT, file)
            ans = ""
            if t == "train":
                ans = r.train_file_path
            elif t == "test":
                ans = r.test_file_path
            elif t == "sample":
                ans = r.sample_file_path
            elif t == "data":
                ans = r.data_file_path
            self.assertEqual(ans, path)

    def test_to_pandas(self):
        r = Resource(DATA_ROOT, ["sentiment", "text"], "sentiment")
        target, text = r.train_data(split_target=True)
        self.assertEqual(len(target), 1)
        self.assertEqual(len(text), 1)

        print(r.train_data().head(1))

    def test_to_indexed(self):
        r = Resource(DATA_ROOT, ["sentiment", "text"], "sentiment")
        r_indexed = r.to_indexed().make_vocab(min_word_count=0)

        self.assertTrue(os.path.exists(r_indexed.vocab_file_path))
        vocab = r_indexed.vocab_data()
        self.assertEqual(len(vocab), 4)  # good/bad/sentence/unk (train + test)
        train_idx = r_indexed.train_data()
        self.assertEqual(len(train_idx), 1)
        os.remove(r_indexed.vocab_file_path)


if __name__ == "__main__":
    unittest.main()
