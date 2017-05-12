import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestNewsGroup20(unittest.TestCase):

    def test_extract(self):
        d = chazutsu.datasets.NewsGroup20()
        file_path = self._download_file(d)
        news_path = d.extract(file_path)

        try:
            with open(news_path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.split("\t")
                    if len(els) != 5:
                        print(els)
                        print(len(els))
                        raise Exception("data file is not constructed by label and text.")
        except Exception as ex:
            if os.path.isfile(file_path):
                os.remove(file_path)
            self.fail(ex)

        if os.path.isfile(file_path):
            os.remove(file_path)
        #os.remove(news_path)

    def test_parse(self):
        d = chazutsu.datasets.NewsGroup20()
        subject, author, text = d.parse(raw_text=sample_text)
        self.assertEqual(subject, "Re: Political Atheists?")
        self.assertEqual(author, "Keith Allan Schneider")
        self.assertTrue(text.startswith("If I"))

    def _download_file(self, dataset):
        url = dataset.download_url
        file_path = os.path.join(DATA_ROOT, "newsgroup20")
        r = requests.get(url)

        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)

        return file_path


sample_text = """
From: keith@cco.caltech.edu (Keith Allan Schneider)
Subject: Re: Political Atheists?

bobbe@vice.ICO.TEK.COM (Robert Beauchaine) writes:

>>If I kill this person [an innocent person convicted of murder],
>>then a murder would be committed, but I would not be the murderer.  At least,
"""


if __name__ == "__main__":
    unittest.main()
