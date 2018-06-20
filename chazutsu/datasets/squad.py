import os
import csv
import json
from collections import Counter

from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class SQuAD(Dataset):

    def __init__(self, kind="train", version="v2.0"):

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

        versions = ("v1.1", "v2.0")

        if version not in ("v1.1", "v2.0"):
            raise Exception(
                "You have to choose version from {}".format(",".join(versions)))

        endpoint = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        urls = {
            "train": endpoint + "train-{}.json".format(version),
            "dev": endpoint + "dev-{}.json".format(version)
        }

        if kind not in urls:
            keys = ",".join(urls.keys())
            raise Exception("You have to choose kind from {}".format(keys))

        self.kind = kind
        self.version = version
        self.download_url = urls[kind]
        self.original_file = os.path.basename(self.download_url)
        self.columns = ["context", "question", "answer", "start", "end"]

    @classmethod
    def train(cls):
        return cls("train")

    @classmethod
    def dev(cls):
        return cls("dev")

    @property
    def root_name(self):
        return self.name.lower().replace(" ", "_") + "_{}_{}".format(self.kind, self.version)

    def download(self,
                 directory="", shuffle=True, test_size=0, sample_count=0,
                 force=False):
        if test_size:
            raise Exception("The dataset is already splitted to train & dev.")

        return super().download(directory, shuffle, 0, sample_count, force)

    def prepare(self, dataset_root, _):
        original_file_path = os.path.join(dataset_root, self.original_file)
        write_file_path = os.path.splitext(original_file_path)[0] + ".txt"
        write_file = open(write_file_path, mode="w", encoding="utf-8", newline="")
        writer = csv.writer(write_file, delimiter="\t")

        self.logger.info("Preprocessing {}".format(original_file_path))
        with open(original_file_path, encoding="utf-8") as rf:
            data = json.load(rf)["data"]

        make_row = getattr(
            self, "make_row_{}".format(self.version.replace(".", "_")))

        for article in xtqdm(data):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].replace("\n", " ")
                for qa in paragraph["qas"]:
                    question = qa["question"].strip().replace("\n", "")
                    row = make_row(context, question, qa)
                    writer.writerow(row)

        self.trush(original_file_path)
        write_file.close()

        return write_file_path

    @staticmethod
    def make_row_v1_1(context, question, qa):
        answers, starts = zip(
            *((a["text"], a["answer_start"]) for a in qa["answers"]))
        spans = [(start, start + len(text))
                 for text, start in zip(answers, starts)]
        most_frequent_span = Counter(spans).most_common(1)[0][0]
        target_index = max(range(len(spans)),
                           key=lambda x: spans[x] == most_frequent_span)
        # pick the most frequent answer
        answer = answers[target_index]
        start, end = most_frequent_span
        return (context, question, start, end, answer)

    @classmethod
    def make_row_v2_0(cls, context, question, qa):
        if not qa["is_impossible"]:
            # You can answer the question
            return cls.make_row_v1_1(context, question, qa)
        else:
            # You can't answer the question
            return (context, question, -1, -1, "")

    def make_resource(self, data_root):
        if self.kind == "train":
            return Resource(data_root, columns=self.columns)
        elif self.kind == "dev":
            return Resource(data_root, columns=self.columns)
