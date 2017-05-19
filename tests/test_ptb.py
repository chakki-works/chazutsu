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
        tokenized, vocab = r.tokenize("valid")
        self.assertTrue(len(tokenized) > 0)

        rev_vocab = {v:k for k, v in vocab.items()}
        print(tokenized[:10])
        print([rev_vocab[i] for i in tokenized[:10]])

    def test_make_vocab(self):
        test_root = os.path.join(DATA_ROOT, "test_make_vocab")
        if not os.path.isdir(test_root):
            os.mkdir(test_root)

        test_file = os.path.join(test_root, "make_vocab.train.txt")
        content = ""
        with open(test_file, mode="w", encoding="utf-8") as f:
            content += " ".join(["apple"] * 10) + "\n"
            content += " ".join(["banana"] * 4) + "\n"
            content += " ".join(["cherry"] * 5) + "\n"
            f.write(content)

        from chazutsu.datasets.ptb import PTBResource
        pr = PTBResource(test_root)

        tokenized, vocab = pr.tokenize(kind="train", min_word_count=5)
        self.assertTrue(len(vocab), 4)  # apple, cherry, <unk>, <eos>
        for v in vocab:
            self.assertTrue(v in ["apple", "cherry", "<unk>", "<eos>"])

        unk_id = vocab["<unk>"]

        rev_vocab = {v:k for k, v in vocab.items()}
        print([rev_vocab[i] for i in tokenized])

        self.assertEqual(len([t for t in tokenized if t == unk_id]), 4)

        shutil.rmtree(test_root)


if __name__ == "__main__":
    unittest.main()
