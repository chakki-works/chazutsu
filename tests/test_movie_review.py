import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestMovieReview(unittest.TestCase):

    def test_extract_polarity(self):
        d = chazutsu.datasets.MovieReview.polarity()

        file_path = self._download_file(d)
        path = d._extract_polarity(file_path)

        pos = 0
        neg = 0

        try:
            with open(path) as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        pos += 1
                    else:
                        neg += 1
        except Exception as ex:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.fail(ex)
        count = d.get_line_count(path)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        os.remove(path)
        # pos=1000, neg=1000
        self.assertEqual(count, 2000)
        self.assertEqual(pos, 1000)
        self.assertEqual(neg, 1000)

    def test_extract_polarity_v1(self):
        d = chazutsu.datasets.MovieReview.polarity_v1()

        file_path = self._download_file(d)
        path = d._extract_polarity_v1(file_path)

        pos = 0
        neg = 0

        try:
            with open(path) as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        pos += 1
                    else:
                        neg += 1
        except Exception as ex:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.fail(ex)
        count = d.get_line_count(path)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        os.remove(path)
        # pos=1000, neg=1000
        self.assertEqual(count, 5331 + 5331)
        self.assertEqual(pos, 5331)
        self.assertEqual(neg, 5331)

    def test_extract_rating(self):
        d = chazutsu.datasets.MovieReview.rating()

        file_path = self._download_file(d)
        path = d._extract_rating(file_path)

        try:
            with open(path) as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
        except Exception as ex:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.fail(ex)

        count = d.get_line_count(path)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        os.remove(path)

        self.assertTrue(count > 0)

    def test_extract_subjectivity(self):
        d = chazutsu.datasets.MovieReview.subjectivity()

        file_path = self._download_file(d)
        path = d._extract_subjectivity(file_path)

        sub = 0
        obj = 0

        try:
            with open(path) as f:
                for ln in f:
                    els = ln.strip().split("\t")
                    if len(els) != 2:
                        raise Exception("data file is not constructed by label and text.")
                    if els[0] == "1":
                        sub += 1
                    else:
                        obj += 1
        except Exception as ex:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.fail(ex)
        count = d.get_line_count(path)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        os.remove(path)
        # sub=5000, obj=5000
        self.assertEqual(count, 5000*2)
        self.assertEqual(sub, 5000)
        self.assertEqual(obj, 5000)
    
    def test_download(self):
        root = chazutsu.datasets.MovieReview.subjectivity().download(DATA_ROOT)
        shutil.rmtree(root)

    def _download_file(self, dataset):
        url = dataset.download_url
        file_path = os.path.join(DATA_ROOT, dataset.kind)
        r = requests.get(url)

        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)

        return file_path


if __name__ == "__main__":
    unittest.main()
