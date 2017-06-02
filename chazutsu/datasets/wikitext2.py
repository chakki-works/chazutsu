import os
import tarfile
import shutil
from collections import Counter
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class WikiText2(Dataset):

    def __init__(self):
        super().__init__(
            name="WikiText-2",
            site_url="https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/",
            download_url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
            description="The dataset for language modeling that is larger than PTB (over 2 times)."
            )
    
    def download(self, directory="", shuffle=False, test_size=0, sample_count=0, keep_raw=False, force=False):
        if test_size != 0:
            raise Exception("This dataset is already splitted to train & test.")
        
        # in language modeling, shuffle is not needed
        return super().download(directory, False, 0, sample_count, keep_raw, force)

    def extract(self, path):
        dir, file_name = os.path.split(path)
        extracteds = self.extract_file(
            path, 
            [
                "wikitext-2/wiki.train.tokens", 
                "wikitext-2/wiki.test.tokens", 
                "wikitext-2/wiki.valid.tokens"
                ],
            remove=True
        )

        train_file_path = os.path.join(dir, "wiki.train.tokens")
        return train_file_path
    
    def make_resource(self, data_root):
        return Resource(
            data_root,
            ["sentence"],
            pattern={
                "train": ".train",
                "test": ".test",
                "valid": ".valid",
                "samples": "_samples"
            }
        )
