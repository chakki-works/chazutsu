import os
import re
import json
from collections import Counter
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource, IndexedResource


class MultiNLI(Dataset):
    LABEL_MAP = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
        "hidden": 0
    }

    def __init__(self, matched=True, full=False):
        """
        matched: matched dataset for mismatched dataset
        full: use full columns in the dataset. If full=False then extract only "gold_label", "genre", "sentence1", "sentence2"
        """
        super().__init__(
            name="MultiNLI",
            site_url="http://www.nyu.edu/projects/bowman/multinli/",
            download_url="https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/multinli_0.9_jsonl.zip",
            description="multi genre's text entailment dataset. There are in-domain(matched) and cross-domain(mismatched) data."
            )
        self.matched = matched
        self.is_full = full
        self.columns = []
        if self.is_full:
            self.columns = ["label", "annotator_labels", "genre", "pairID", "sentence1", "sentence2", "sentence1_parse", "sentence2_parse"]
        else:
            self.columns = ["label", "genre", "pairID", "sentence1", "sentence2"]
        self._tokenize_pattern = re.compile("\(|\)")
    
    def download(self, directory="", shuffle=True, test_size=0, sample_count=0, keep_raw=False, force=False):
        if test_size != 0:
            raise Exception("This dataset is already splitted to train & test.")
        # in language modeling, shuffle is not needed
        return super().download(directory, shuffle, 0, sample_count, keep_raw, force)

    @classmethod
    def matched(cls, full=False):
        return MultiNLI(True, full)

    @classmethod
    def mismatched(cls, full=False):
        return MultiNLI(False, full)

    def extract(self, path):
        dir, file_name = os.path.split(path)
        kind = "matched" if self.matched else "mismatched"
        extracteds = self.extract_file(
            path, 
            [
                "multinli_0.9_jsonl/multinli_0.9_train.jsonl", 
                "multinli_0.9_jsonl/multinli_0.9_{}_dev.jsonl".format(kind),
                "multinli_0.9_jsonl/multinli_0.9_{}_unlabeled_test.jsonl".format(kind)
            ],
            remove=True
        )

        train_file = ""
        for e in extracteds:
            preprocessed = self.preprocess_file(e)
            os.remove(e)
            if "_train" in preprocessed:
                train_file = preprocessed

        return preprocessed
    
    def preprocess_file(self, path):
        write_file_path = path.replace(".jsonl", ".txt")
        write_file = open(write_file_path, mode="w", encoding="utf-8")
        file_kind = path.split("_")[-1]

        self.logger.info("Preprocessing {} file".format(file_kind))
        total_count = self.get_line_count(path)
        with open(path, encoding="utf-8") as rf:
            for line in xtqdm(rf, total=total_count):
                preprocessed = self.preprocess_jsonl(line)
                if preprocessed:
                    w_line = "\t".join(preprocessed) + "\n"
                    write_file.write(w_line)
        
        write_file.close()
        return write_file_path

    def preprocess_jsonl(self, jsonl):
        loaded = json.loads(jsonl)
        if loaded["gold_label"] not in self.LABEL_MAP:
            return None

        loaded["label"] = str(self.LABEL_MAP[loaded["gold_label"]])
        if "genre" not in loaded or not loaded["genre"]:
            loaded["genre"] = "-"
        loaded["sentence1"] = self._tokenized_str(loaded["sentence1_binary_parse"])
        loaded["sentence2"] = self._tokenized_str(loaded["sentence2_binary_parse"])

        values = [loaded[c] for c in self.columns]
        return values

    def _tokenized_str(self, sentence):
        _s = self._tokenize_pattern.sub("", sentence)
        tokens = _s.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        return " ".join(tokens)

    def make_resource(self, data_root):
        return NLIResource(data_root, columns=self.columns, target="label")


class NLIResource(Resource):

    def __init__(self, 
        root,
        columns=None,
        target="",
        separator="\t",
        pattern=()):

        super().__init__(
            root, 
            columns,
            target,
            separator,
            {
                "train": "_train",
                "test": "_test",
                "dev": "_dev",
                "sample": "_samples"
            })
    
    @property
    def dev_file_path(self):
        return self._get_prop("dev")
    
    def dev_data(self, split_target=False):
        return self._get_data("dev", split_target)

    def to_indexed(self, vocab_resources=("train", "dev", "test"), vocab_columns=("sentence1", "sentence2")):
        ir = IndexedResource(self, vocab_resources, vocab_columns)
        return ir
