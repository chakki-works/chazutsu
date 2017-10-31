import os
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class Text8(Dataset):

    def __init__(self, kind="en"):
        super().__init__(
            name="Text8",
            site_url="http://mattmahoney.net/dc/textdata",
            download_url="http://mattmahoney.net/dc/text8.zip",
            description="This dataset offers cleaned English Wikipedia text"
            )
        self.kind = kind
        if self.kind == "ja":
            self.site_url = "https://github.com/Hironsan/ja.text8"
            self.download_url = "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/ja.text8.zip"  # noqa
            self.description = self.description.replace("English", "Japanese")

    @classmethod
    def en(cls):
        return Text8("en")

    @classmethod
    def ja(cls):
        return Text8("ja")

    def download(self,
                 directory="", shuffle=True, test_size=10, sample_count=0,
                 force=False):
        if test_size < 1:
            raise Exception("This dataset is splitted by byte. test_size "
                            "means how many mega bytes are used for test.")

        self._test_size = test_size
        return super().download(directory, False, 0, sample_count, force)

    @property
    def extract_targets(self):
        inner_file = "text8" if self.kind == "en" else "ja.text8"
        return [inner_file]

    def prepare(self, dataset_root, extracted_path):
        inner_file = self.extract_targets[0]
        target = "text8" if self.kind == "en" else "text8ja"
        train_file_path = os.path.join(dataset_root, target + ".train.txt")
        if self._test_size == 0:
            os.rename(os.path.join(dataset_root, inner_file), train_file_path)
        else:
            test_file_path = os.path.join(dataset_root, target + ".test.txt")

            line = ""
            test_byte = self._test_size * 1000000
            with open(os.path.join(extracted_path, inner_file),
                      encoding="utf-8") as f:
                line = f.readline().strip()
            with open(train_file_path, mode="w", encoding="utf-8") as train:
                train.write(line[:-test_byte])
            with open(test_file_path, mode="w", encoding="utf-8") as test:
                test.write(line[-test_byte:])

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
