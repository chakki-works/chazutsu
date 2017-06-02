import os
import shutil
from collections import Counter
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class Text8(Dataset):

    def __init__(self):
        super().__init__(
            name="Text8",
            site_url="http://mattmahoney.net/dc/textdata",
            download_url="http://mattmahoney.net/dc/text8.zip",
            description="This dataset offers cleaned English Wikipedia text"
            )
        self._test_size = 0
    
    def download(self, directory="", shuffle=False, test_size=10, sample_count=0, keep_raw=False, force=False):
        if test_size < 1:
            raise Exception("This dataset is splitted by byte. test_size means how many mega bytes are used for test.")
        
        self._test_size = test_size
        return super().download(directory, False, 0, sample_count, keep_raw, force)

    def extract(self, path):
        dir, file_name = os.path.split(path)
        extracteds = self.extract_file(path, ["text8"], remove=True)

        train_file_path = os.path.join(dir, "text8.train.txt")
        if self._test_size == 0:
            os.rename(os.path.join(dir, "text8"), train_file_path)
        else:
            test_file_path = os.path.join(dir, "text8.test.txt")

            line = ""
            test_byte = self._test_size * 1000000
            with open(os.path.join(dir, "text8"), encoding="utf-8") as f:
                line = f.readline().strip()
            with open(train_file_path, mode="w", encoding="utf-8") as train:
                train.write(line[:-test_byte])
            with open(test_file_path, mode="w", encoding="utf-8") as test:
                test.write(line[-test_byte:])
        
            os.remove(os.path.join(dir, "text8"))

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
