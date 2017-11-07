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
                one = "\t".join(["o", "good apple is delicious.", "1"])
                two = "\t".join(["x", "bad bananas taste bad.", "-1"])
                three = "\t".join(["x", "bad cherries are yellow .", "-1"])
                four = "\t".join(["-", "normal fruits have flavor.", "0"])
                content = "\n".join([one, two, three, four])
            elif t == "test":
                one = "\t".join(["o", "good grape is delicious.", "1"])
                two = "\t".join(["x", "bad oranges taste bad.", "-1"])
                three = "\t".join(["x", "bad pears are yellow .", "-1"])
                content = "\n".join([one, two, three])
            elif t == "sample":
                content = "\t".join(["x", "bad cherries are yellow .", "-1"])
            else:
                content = "\t".join(["-", "normal fruits have flavor.", "0"])
            
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
        self.assertEqual(len(target), 4)
        self.assertEqual(len(text), 4)

        print(r.train_data().head(1))

    def test_to_batch(self):
        r = Resource(DATA_ROOT, ["sentiment", "text", "score"], "sentiment")
        X, y = r.to_batch("train")
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4, 1))
        r.make_vocab()
        r.column("text").as_word_seq(fixed_len=5)
        X, y = r.to_batch("train", columns=("sentiment", "text"))
        self.assertEqual(X.shape, (4, 5, len(r.vocab)))

    def test_to_batch_iter(self):
        r = Resource(DATA_ROOT, ["sentiment", "text", "score"], "sentiment")
        r.make_vocab()
        batch_size = 2
        fixed_len = 5
        r.column("text").as_word_seq(fixed_len=fixed_len)
        iterator, count = r.to_batch_iter(
                            "train", columns=("sentiment", "text"),
                            batch_size=batch_size)
        self.assertEqual(count, batch_size)
        for i in range(4):
            X, y = next(iterator)
            self.assertEqual(y.shape, (batch_size, 1))
            self.assertEqual(X.shape, (batch_size, fixed_len, len(r.vocab)))
            print(r.column("text").back(X))


if __name__ == "__main__":
    unittest.main()
