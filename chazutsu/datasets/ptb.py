import os
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class PTB(Dataset):

    def __init__(self):
        super().__init__(
            name="Penn Tree Bank",
            site_url="https://github.com/tomsercu/lstm",
            download_url="https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/ptb.zip",  # noqa
            description="basic language modeling dataset that omits linguistic structure annotations."  # noqa
            )

    def download(self,
                 directory="", shuffle=True, test_size=0, sample_count=0,
                 force=False):
        if test_size != 0:
            raise Exception("The dataset is already splitted to train & test.")
        # in language modeling, shuffle is not needed
        return super().download(directory, False, 0, sample_count, force)

    def extract(self, path):
        dir, file_name = os.path.split(path)
        self.extract_file(
            path,
            ["ptb/ptb.train.txt", "ptb/ptb.valid.txt", "ptb/ptb.test.txt"]
        )
        train_file_path = os.path.join(dir, "ptb.train.txt")

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
