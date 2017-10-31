import os
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class Gigaword(Dataset):

    def __init__(self):

        super().__init__(
            name="Gigaword",
            site_url="https://catalog.ldc.upenn.edu/ldc2003t05",
            download_url="https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/Gigaword.zip",  # noqa
            description="Gigaword is news text corpora con. " \
                        "This dataset is preprocessed version that is used at " \
                        "'A Neural Attention Model for Sentence Summarization'."
            )

    @property
    def extract_targets(self):
        return ["Gigaword/input.txt",
                "Gigaword/task1_ref0.txt"]

    def prepare(self, dataset_root, extracted_path):
        news_path = self.__get_extracted_path(extracted_path, 0)
        summary_path = self.__get_extracted_path(extracted_path, 1)
        write_file_name = "gigaword.txt"
        write_path = os.path.join(dataset_root, write_file_name)

        write_file = open(write_path, "w", encoding="utf-8")
        with open(news_path, encoding="utf-8") as n:
            with open(summary_path, encoding="utf-8") as s:
                for nline, sline in zip(n, s):
                    line = "\t".join([
                        nline.strip(), sline.strip()
                        ]) + "\n"
                    write_file.write(line)
        write_file.close()
        return write_path

    def __get_extracted_path(self, extracted_path, index):
        return os.path.join(
                extracted_path, 
                os.path.basename(self.extract_targets[index]))

    def make_resource(self, data_root):
        return Resource(
                data_root,
                columns=["news", "summary"],
                target="summary")
