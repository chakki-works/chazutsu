import os
import pandas as pd
from chazutsu.datasets.framework.vocabulary import Vocabulary


class Resource():

    def __init__(self, 
        root,
        columns=None,
        target="",
        separator="\t",
        pattern=()):

        self.root = root
        self.columns = columns
        self.target = target
        self.separator = separator
        self._pattern = pattern
        if len(self._pattern) == 0:
            self._pattern = {
                "train": "_train",
                "test": "_test",
                "valid": "_valid",
                "sample": "_samples"
            }
        self._resource_name = ""
        self._resource = {}

        if target and target not in columns:
            raise Exception("Target have to be selected from columns.")
        self.find_resource()
    
    def find_resource(self):
        for f in os.listdir(self.root):
            p = os.path.join(self.root, f)
            if not os.path.isfile(p) or f.startswith("."):
                continue
            
            n, e = os.path.splitext(f)
            if e == ".vocab":
                continue  # skip vocab file

            match = False
            for kind in self._pattern:
                if n.endswith(self._pattern[kind]):
                    self._resource[kind] = p
                    self._resource_name = n.replace(self._pattern[kind], "")
                    match = True
                    break

            if not match:
                self._resource["data"] = p
                self._resource_name = n
        
        if "data" not in self._resource:
            self._resource["data"] = self._resource["train"]

    @property
    def data_file_path(self):
        return self._get_prop("data")

    @property
    def train_file_path(self):
        return self._get_prop("train")

    @property
    def test_file_path(self):
        return self._get_prop("test")

    @property
    def valid_file_path(self):
        return self._get_prop("valid")

    @property
    def sample_file_path(self):
        return self._get_prop("sample")
    
    def _get_prop(self, name):
        return "" if name not in self._resource else self._resource[name]        

    def data(self, split_target=False):
        return self._get_data("data", split_target)
    
    def train_data(self, split_target=False):
        return self._get_data("train", split_target)

    def test_data(self, split_target=False):
        return self._get_data("test", split_target)

    def valid_data(self, split_target=False):
        return self._get_data("valid", split_target)

    def sample_data(self, split_target=False):
        return self._get_data("sample", split_target)

    def _get_data(self, name, split_target):
        if name not in self._resource:
            raise Exception("Can not find {} data resource.".format(name))
        return self._to_pandas(self._resource[name], split_target)

    def _to_pandas(self, path, split_target):
        df = pd.read_table(path, header=None, names=self.columns)

        if not split_target:
            return df
        elif self.target and self.target in df.columns:
            target = df[self.target]
            df.drop(self.target, axis=1, inplace=True)
            return target, df
        else:
            return df
    
    def to_indexed(self, vocab_resources=("train", "valid", "test"), vocab_columns=()):
        ir = IndexedResource(self, vocab_resources, vocab_columns)
        return ir


class IndexedResource(Resource):

    def __init__(self, 
        resource,
        vocab_resources=("train", "valid", "test"),
        vocab_columns=()):

        if len(resource._resource) == 0:
            raise Exception("supplied resource does not have any resource.")

        super().__init__(
            resource.root,
            resource.columns,
            resource.target,
            resource.separator,
            resource._pattern
        )
        self.vocab_resources = vocab_resources
        self.vocab_columns = vocab_columns
        if len(self.vocab_columns) == 0:
            self.vocab_columns = [self.columns[-1]]  # assume last column would be text

        self._resource_name = resource._resource_name
        self._original_resource = resource._resource
        self.vocab = Vocabulary(self.root, self._resource_name)

    def make_vocab(self, 
        tokenizer=None, 
        vocab_size=-1,
        min_word_count=0, 
        end_of_sentence="", 
        unknown="<unk>",
        reserved_words=(),
        force=False):
        
        if self.vocab.has_vocab() and not force:
            return self

        if tokenizer is not None:
            self.vocab.tokenizer = tokenizer
        if end_of_sentence:
            self.vocab.end_of_sentence = end_of_sentence
        if unknown:
            self.vocab.unknown = unknown

        paths = []
        for kind in self._original_resource:
            p = self._original_resource[kind]
            if p and kind in self.vocab_resources:
                paths.append(p)
        column_indexes = [i for i, c in enumerate(self.columns) if c in self.vocab_columns]

        if len(paths) > 0:
            self.vocab.make(paths, vocab_size, min_word_count, column_indexes, self.separator, reserved_words)
        
        return self

    @property
    def vocab_file_path(self):
        if self.vocab.has_vocab():
            return self.vocab._vocab_file_path
        else:
            return ""

    def vocab_data(self):
        v = self.vocab._vocab
        if len(v) == 0 and self.vocab.has_vocab():
            self.vocab.load()
            v = self.vocab._vocab        
        return v

    def str_to_ids(self, sentence):
        return self.vocab.str_to_ids(sentence)

    def ids_to_words(self, ids):
        return self.vocab.ids_to_words(ids)

    def ids_to_one_hots(self, ids):
        return self.vocab.ids_to_one_hots(ids)

    def _to_pandas(self, path, split_target):
        if not self.vocab.has_vocab():
            raise Exception("You have to make vocabulary by make_vocab first.")

        df = pd.read_table(path, header=None, names=self.columns)
        for c in self.vocab_columns:
            df[c] = df[c].map(self.vocab.str_to_ids)

        if not split_target:
            return df
        elif self.target and self.target in df.columns:
            target = df[self.target]
            df.drop(self.target, axis=1, inplace=True)
            return target, df
        else:
            return df
