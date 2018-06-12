import os
import json

from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class SQuAD(Dataset):

    def __init__(self, kind="train"):

        super().__init__(
            name="SQuAD",
            site_url="https://rajpurkar.github.io/SQuAD-explorer/",
            download_url="",
            description="Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset,"
            "consisting of questions posed by crowdworkers on a set of Wikipedia articles, "
            "where the answer to every question is a segment of text, or span, from the corresponding reading passage."
            "With 100,000+ question-answer pairs on 500+ articles,"
            "SQuAD is significantly larger than previous reading comprehension datasets."
        )

        urls = {
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
            "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        }

        if kind not in urls:
            keys = ",".join(urls.keys())
            raise Exception("You have to choose kind from {}".format(keys))

        self.kind = kind
        self.download_url = urls[kind]
        self.original_file = "train-v1.1.json" if kind == "train" else "dev-v1.1.json"
        self.columns = ["context", "question", "answer", "start", "end"]

    @classmethod
    def train(cls):
        return cls("train")

    @classmethod
    def dev(cls):
        return cls("dev")

    @property
    def root_name(self):
        return self.name.lower().replace(" ", "_") + "_" + self.kind

    def download(self,
                 directory="", shuffle=True, test_size=0, sample_count=0,
                 force=False):
        if test_size:
            raise Exception("The dataset is already splitted to train & dev.")

        return super().download(directory, shuffle, 0, sample_count, force)

    def prepare(self, dataset_root, _):
        original_file_path = os.path.join(dataset_root, self.original_file)
        write_file_path = os.path.splitext(original_file_path)[0] + ".txt"
        write_file = open(write_file_path, mode="w", encoding="utf-8")

        self.logger.info("Preprocessing {}".format(original_file_path))
        with open(original_file_path, encoding="utf-8") as rf:
            data = json.load(rf)["data"]

        for article in xtqdm(data):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    for a in qa["answers"]:
                        text = a["text"]
                        start = a["answer_start"]
                        end = start + len(text.split())
                        line = "{}\t{}\t{}\t{}\t{}\n".format(
                            context, question, text, start, end)
                        write_file.write(line)
        self.trush(original_file_path)

        return write_file_path

    def make_resource(self, data_root):
        if self.kind == "train":
            return Resource(data_root, columns=self.columns)
        elif self.kind == "dev":
            return Resource(data_root, columns=self.columns)
