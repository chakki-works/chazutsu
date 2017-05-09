import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
from chazutsu.datasets.framework.dataset import Dataset


class SampleDataset(Dataset):

    def __init__(self):
        super().__init__(
            name="sample_dataset",
            site_url="https://github.com/chakki-works/chazutsu",
            download_url="https://github.com/chakki-works/chazutsu/archive/master.zip",
            description="sample dataset"
            )

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

class TestDataset(unittest.TestCase):

    def test_save_dataset(self):
        d = SampleDataset()
        path = d.save_dataset(DATA_ROOT)
        self.assertTrue(path)
        os.remove(path)
    
    def test_extract_file_zip(self):
        d = SampleDataset()
        path = d.save_dataset(DATA_ROOT)
        extracteds = d.extract_file(path, ["README.md", "docs/chazutsu.png"])
        self.assertEqual(len(extracteds), 2)
        for e in extracteds:
            os.remove(e)

    def test_extract_file_tar_gz(self):
        d = SampleDataset()
        d.download_url = d.download_url.replace(".zip", ".tar.gz")
        path = d.save_dataset(DATA_ROOT)
        extracteds = d.extract_file(path, ["LICENSE", "docs/feature.png"])
        self.assertEqual(len(extracteds), 2)
        for e in extracteds:
            os.remove(e)
        

if __name__ == "__main__":
    unittest.main()
