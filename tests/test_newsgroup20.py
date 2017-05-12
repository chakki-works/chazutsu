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
        r = chazutsu.datasets.NewsGroup20().download(directory=DATA_ROOT, test_size=0)

        try:
            with open(r.path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.split("\t")
                    if len(els) != 5:
                        print(els)
                        print(len(els))
                        raise Exception("data file is not constructed by label and text.")
 
        except Exception as ex:
            if os.path.isfile(r.path):
                os.remove(r.path)
            self.fail(ex)
        
        self.assertTrue(len(r.data().columns), 5)

        if os.path.isfile(r.path):
            os.remove(r.path)
        shutil.rmtree(r.root)

    def test_parse(self):
        d = chazutsu.datasets.NewsGroup20()
        subject, author, text = d.parse(raw_text=sample_text)
        self.assertEqual(subject, "Re: Political Atheists?")
        self.assertEqual(author, "Keith Allan Schneider")
        self.assertTrue(text.startswith("If I"))


sample_text = """
From: keith@cco.caltech.edu (Keith Allan Schneider)
Subject: Re: Political Atheists?

bobbe@vice.ICO.TEK.COM (Robert Beauchaine) writes:

>>If I kill this person [an innocent person convicted of murder],
>>then a murder would be committed, but I would not be the murderer.  At least,
"""


if __name__ == "__main__":
    unittest.main()
