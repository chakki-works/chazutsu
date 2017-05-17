import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import requests
import chazutsu.datasets


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestReutersNews(unittest.TestCase):

    def test_topics(self):
        r = chazutsu.datasets.ReutersNews(kind="topics").download(directory=DATA_ROOT, force=True)
        self.assertTrue(len(r.data().columns), 3)
        print(r.data().columns)
        shutil.rmtree(r.root)

    def test_industries(self):
        r = chazutsu.datasets.ReutersNews().industries().download(directory=DATA_ROOT, force=True)
        self.assertTrue(len(r.data().columns), 3)
        print(r.data().columns)
        shutil.rmtree(r.root)

    def test_regions(self):
        r = chazutsu.datasets.ReutersNews().regions().download(directory=DATA_ROOT, force=True)
        self.assertTrue(len(r.data().columns), 2)
        print(r.data().columns)
        shutil.rmtree(r.root)


if __name__ == "__main__":
    unittest.main()
