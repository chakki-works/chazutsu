import os
import sys
import shutil
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data/mr")
if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)


class TestMovieReview(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)

    def test_prepare_polarity(self):
        d = chazutsu.datasets.MovieReview.polarity()
        dataset_root, extracted = d.save_and_extract(DATA_ROOT)
        path = d.prepare(dataset_root, extracted)

        pos = 0
        neg = 0

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        pos += 1
                    else:
                        neg += 1
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)
        count = d.get_line_count(path)
        
        d.clear_trush()
        os.remove(path)
        # pos=1000, neg=1000
        self.assertEqual(count, 2000)
        self.assertEqual(pos, 1000)
        self.assertEqual(neg, 1000)

    def test_extract_polarity_v1(self):
        d = chazutsu.datasets.MovieReview.polarity_v1()
        dataset_root, extracted = d.save_and_extract(DATA_ROOT)
        path = d.prepare(dataset_root, extracted)

        pos = 0
        neg = 0

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        pos += 1
                    else:
                        neg += 1
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)
        count = d.get_line_count(path)

        d.clear_trush()
        # pos=1000, neg=1000
        self.assertEqual(count, 5331 + 5331)
        self.assertEqual(pos, 5331)
        self.assertEqual(neg, 5331)

    def test_extract_rating(self):
        d = chazutsu.datasets.MovieReview.rating()
        dataset_root, extracted = d.save_and_extract(DATA_ROOT)
        path = d.prepare(dataset_root, extracted)

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)

        count = d.get_line_count(path)

        d.clear_trush()
        self.assertTrue(count > 0)

    def test_extract_subjectivity(self):
        d = chazutsu.datasets.MovieReview.subjectivity()
        dataset_root, extracted = d.save_and_extract(DATA_ROOT)
        path = d.prepare(dataset_root, extracted)

        sub = 0
        obj = 0

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        sub += 1
                    else:
                        obj += 1
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)
        count = d.get_line_count(path)

        d.clear_trush()
        # sub=5000, obj=5000
        self.assertEqual(count, 5000*2)
        self.assertEqual(sub, 5000)
        self.assertEqual(obj, 5000)

    def test_download(self):
        r = chazutsu.datasets.MovieReview.subjectivity().download(DATA_ROOT)
        target, data = r.test_data(split_target=True)
        self.assertEqual(target.shape[0], data.shape[0])

        r.make_vocab(vocab_size=1000)
        X, y = r.column("review").as_word_seq(fixed_len=20).to_batch("train", with_target=True)
        self.assertEqual(y.shape, (len(y), 1))
        self.assertEqual(X.shape, (len(y), 20, len(r.vocab)))

        backed = r.column("review").back(X)
        print(backed[:3])
        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
