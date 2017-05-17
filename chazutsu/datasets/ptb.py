import os
import tarfile
import shutil
from collections import Counter
from tqdm import tqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class PTB(Dataset):

    def __init__(self):
        super().__init__(
            name="Penn Tree Bank",
            site_url="https://github.com/tomsercu/lstm",
            download_url="https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/ptb.zip",
            description="basic language modeling dataset that omits linguistic structure annotations."
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
            ["ptb/ptb.train.txt", "ptb/ptb.valid.txt", "ptb/ptb.test.txt"],
            remove=True
        )
        train_file_path = os.path.join(dir, "ptb.train.txt")

        return train_file_path
    
    def make_resource(self, data_root):
        return PTBResource(data_root)    


class PTBResource(Resource):

    def __init__(self,
        root,
        train_file_suffix=".train",
        test_file_suffix=".test",
        sample_file_suffix="_samples"):

        super().__init__(
            root, 
            ["sentence"],
            "",
            train_file_suffix, 
            test_file_suffix, 
            sample_file_suffix)
        
        self.valid_file_path = ""
        self.vocab_file_path = {
            "train": "",
            "valid": "",
            "test": ""
        }

        self.path = self.train_file_path
        for f in os.listdir(self.root):
            p = os.path.join(self.root, f)
            n, e = os.path.splitext(f)
            if n.endswith(".valid"):
                self.valid_file_path = p
            if n.endswith(".vocab"):
                if "train" in n:
                    self.self.vocab_file_path["train"] = p
                elif "valid" in n:
                    self.self.vocab_file_path["valid"] = p
                elif "test" in n:
                    self.self.vocab_file_path["test"] = p
    
    def validation_data(self):
        return self._to_pandas(self.valid_file_path, split_target)
    
    def make_vocab(self, kind="train"):
        if kind not in self.vocab_file_path:
            raise Exception("The data kind is not correct (select from train/valid/test).")

        path = self.get_file_path(kind)
        dir, file_name = os.path.split(path)
        vocab_path = os.path.join(dir, kind + ".vocab")

        vocab = Counter()
        for words in self._read_data_file(path):
            for w in words:
                vocab[w] += 1
        
        _vocab = [k_v[0] for k_v in vocab.most_common()]
        if "<unk>" not in _vocab:
            _vocab.append("<unk>")

        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_vocab))
        self.vocab_file_path[kind] = vocab_path
    
    def _read_data_file(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if ln:
                    words = ln.split()
                    words.append("<eos>")
                    yield words

    def get_file_path(self, kind):
        path = ""
        if kind == "train":
            path = self.train_file_path
        elif kind == "test":
            path = self.test_file_path
        elif kind == "valid":
            path = self.valid_file_path
        return path

    def tokenize(self, kind="train", vocab_size=-1):
        if kind not in self.vocab_file_path:
            raise Exception("The data kind is not correct (select from train/valid/test).")
        elif not self.vocab_file_path[kind]:
            self.make_vocab(kind)
        
        vocab = {}
        with open(self.vocab_file_path[kind], encoding="utf-8") as f:
            lines = f.readlines()
            if vocab_size > 0:
                lines = lines[:vocab_size]
            for i, ln in enumerate(lines):
                w = ln.strip()
                vocab[w] = i

        path = self.get_file_path(kind)
        tokenized = []
        for words in self._read_data_file(path):
            for w in words:
                if w in vocab:
                    tokenized.append(vocab[w])
                else:
                    tokenized.append(vocab["<unk>"])

        return tokenized, vocab
