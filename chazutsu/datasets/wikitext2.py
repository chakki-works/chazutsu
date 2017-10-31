import os
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class WikiText2(Dataset):

    def __init__(self):
        super().__init__(
            name="WikiText-2",
            site_url="https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/",  # noqa
            download_url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",  # noqa
            description="The dataset for language modeling that is larger than PTB (over 2 times)."  # noqa
            )

    def download(self,
                 directory="", shuffle=True, test_size=0, sample_count=0,
                 force=False):
        if test_size != 0:
            raise Exception("The dataset is already splitted to train & test.")

        # in language modeling, shuffle is not needed
        return super().download(directory, False, 0, sample_count, force)

    @property
    def extract_targets(self):
        return [
                "wikitext-2/wiki.train.tokens",
                "wikitext-2/wiki.test.tokens",
                "wikitext-2/wiki.valid.tokens"
                ]

    def prepare(self, dataset_root, extracted_path):
        self.move_extracteds(dataset_root, extracted_path)
        train_file_path = os.path.join(dataset_root, "wiki.train.tokens")
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
