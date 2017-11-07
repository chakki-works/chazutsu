import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from chazutsu.datasets.framework.vocabulary import Vocabulary


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestVocabulary(unittest.TestCase):
    TEST_FILE = os.path.join(DATA_ROOT, "vocab_test.txt")

    @classmethod
    def setUpClass(cls):
        line1 = "apple is so sweet fruit!"
        line2 = "Oh apple apple so so so so sweet..."  # I'm calm. It's for test of word count.
        line3 = "Is there any fruit that taste better than apple? No way."
        doc = "\n".join([line1, line2, line3])
        with open(cls.TEST_FILE, mode="wb") as f:
            f.write(doc.encode("utf-8"))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.TEST_FILE):
            os.remove(cls.TEST_FILE)

    def test_make(self):
        vocab = Vocabulary(DATA_ROOT, "test_vocab", end_of_sentence="<eos>")
        vocab.make(self.TEST_FILE, min_word_freq=3)
        self.assertTrue(len(vocab._vocab) == 5)  # apple & so & eos/unk/pad
        
        vocab._vocab = {}
        ids = vocab.str_to_ids("apple so sweet")
        self.assertEqual(ids, [vocab._vocab["apple"], vocab._vocab["so"], vocab._vocab["<unk>"]])

        words = vocab.ids_to_words(ids)
        self.assertEqual(" ".join(words), "apple so <unk>")

        os.remove(vocab._vocab_file_path)

    def test_vocab_size(self):
        vocab = Vocabulary(DATA_ROOT, "test_vocab", end_of_sentence="<eos>")
        vocab.make(self.TEST_FILE, vocab_size=4)
        self.assertEqual(len(vocab._vocab), 4)
        os.remove(vocab._vocab_file_path)

    def test_str_to_matrix(self):
        vocab = Vocabulary(DATA_ROOT, "test_vocab", end_of_sentence="<eos>")
        vocab.make(self.TEST_FILE)
        one_hot = vocab.str_to_matrix("apple so sweet", fixed_len=4)
        self.assertEqual(one_hot.shape, (4, len(vocab)))
        words = vocab.matrix_to_words(one_hot, ignore_padding=True)
        self.assertEqual(len(words), 3)
        os.remove(vocab._vocab_file_path)


if __name__ == "__main__":
    unittest.main()
