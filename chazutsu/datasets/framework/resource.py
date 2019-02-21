import os
import mmap
import numpy as np
import pandas as pd
from chazutsu.datasets.framework.vocabulary import Vocabulary
from chazutsu.datasets.framework import converter as cv


class Resource():

    def __init__(self,
                 root,
                 columns=None, target="",
                 separator="\t", pattern=()):

        self.root = root
        self.columns = columns
        self.target = target
        self.separator = separator
        self._pattern = pattern
        self._vocab = None
        if len(self._pattern) == 0:
            self._pattern = {
                "train": "_train",
                "test": "_test",
                "valid": "_valid",
                "sample": "_samples"
            }
        self._resource_name = ""
        self._resource = {}
        self._batch_def = {}

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

        if "data" not in self._resource and "train" in self._resource:
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

    def _get_data(self, kind, split_target):
        if kind not in self._resource:
            raise Exception("Can not find {} data self.".format(kind))
        return self._to_pandas(self._resource[kind], split_target)

    def _to_pandas(self, path, split_target):
        df = pd.read_csv(path, header=None, names=self.columns, sep="\t")

        if not split_target:
            return df
        elif self.target and self.target in df.columns:
            target = df[self.target]
            df.drop(self.target, axis=1, inplace=True)
            return target, df
        else:
            return df

    def make_vocab(self,
                   vocab_resources=("train", "valid", "test"),
                   columns_for_vocab=(),
                   tokenizer=None,
                   vocab_size=-1,
                   min_word_freq=0,
                   unknown="<unk>",
                   padding="<pad>",
                   end_of_sentence="",
                   reserved_words=("<pad>", "<unk>"),
                   force=False):

        if len(self._resource) == 0:
            raise Exception("Supplied resource does not have any resource.")

        # Prepare the file paths to make vocab
        resource_for_vocab = []
        for kind in vocab_resources:
            if kind in self._resource and self._resource[kind]:
                resource_for_vocab.append(self._resource[kind])
        if len(resource_for_vocab) == 0:
            raise Exception(
                "There are no resource in {} to make vocaburaly.".format(
                    self._resource_name))

        # Prepare the column index to make vocab
        _c_for_vocab = columns_for_vocab
        if len(_c_for_vocab) == 0:
            _c_for_vocab = self.columns  # all columns is used

        _c_for_vocab_idx = [i for i, c in enumerate(self.columns)
                            if c in _c_for_vocab]

        self._vocab = Vocabulary(
            self.root, self._resource_name,
            tokenizer=tokenizer, unknown=unknown, padding=padding,
            end_of_sentence=end_of_sentence)

        if self._vocab.has_vocab() and not force:
            self._vocab.load()
        else:
            self._vocab.make(
                path_or_paths=resource_for_vocab,
                vocab_size=vocab_size, min_word_freq=min_word_freq,
                separator=self.separator, reserved_words=reserved_words,
                target_column_indexes=_c_for_vocab_idx)
        return self._vocab

    @property
    def vocab_file_path(self):
        if self._vocab.has_vocab():
            return self._vocab._vocab_file_path
        else:
            return ""

    @property
    def vocab(self):
        v = self._vocab._vocab
        return v

    def column(self, column):
        return Route(self, column)

    def to_batch(self, kind, columns=()):
        df = self._get_data(kind, split_target=False)
        y = None
        Xs = []
        _columns = columns if len(columns) > 0 else self.columns
        for c in _columns:
            if c not in df.columns:
                raise Exception("{} is not exist.".format(c))
            s = self._to_array(c, df[c])
            if c == self.target:
                y = s
            else:
                Xs.append(s)
        if len(Xs) == 1:
            X = Xs[0]
        else:
            X = np.hstack(Xs)

        return X, y

    def _to_array(self, column, data):
        _cv = cv.Converter()
        if column in self._batch_def:
            _cv = self._batch_def[column]

        s = _cv.flow(data)
        if len(s.shape) == 1:
            s = np.reshape(s, (-1, 1))  # vector to vertical mx
        return s

    def get_line_count(self, kind):
        file_path = self._get_prop(kind)
        count = 0
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    def to_batch_iter(self, kind, batch_size, columns=()):
        iterator = self._to_batch_iter(kind, batch_size, columns)
        batch_count = self.get_line_count(kind) // batch_size
        return iterator, batch_count

    def _to_batch_iter(self, kind, batch_size, columns=()):
        file_path = self._get_prop(kind)
        targets = columns if len(columns) > 0 else self.columns
        while True:
            Xs = {}
            y = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split(self.separator)
                    for i, c in enumerate(self.columns):
                        if c not in targets:
                            continue
                        if c == self.target:
                            y.append(tokens[i])
                        else:
                            if c not in Xs:
                                Xs[c] = []
                            Xs[c].append(tokens[i])
                    if len(y) == batch_size:
                        yb = self._to_array(self.target, y)
                        Xb = []
                        for c in Xs:
                            Xb.append(self._to_array(c, Xs[c]))

                        if len(Xb) == 1:
                            Xb = Xb[0]
                        else:
                            Xb = np.hstack(Xb)
                        yield Xb, yb

                        y.clear()
                        for x in Xs:
                            Xs[x].clear()


class Route():

    def __init__(self, resource, column):
        self.r = resource
        self.column = column

    def as_word_seq(self, fixed_len=-1):
        c = cv.VocabularyConverter(self.r._vocab, fixed_len)
        self.r._batch_def[self.column] = c
        return self

    def as_category(self, labels=(), distincts=False, num_class=-1):
        c = cv.OneHotConverter(labels, distincts, num_class)
        self.r._batch_def[self.column] = c
        return self

    def as_dictid(self, labels=(), distincts=False, num_class=-1):
        c = cv.DictionalyConverter(labels, distincts, num_class)
        self.r._batch_def[self.column] = c
        return self

    def flow(self, series):
        if self.column in self.r._batch_def:
            return self.r._batch_def[self.column].flow(series)
        else:
            return None

    def back(self, converted):
        if self.column in self.r._batch_def:
            return self.r._batch_def[self.column].back(converted)
        else:
            return None

    def to_batch(self, kind, with_target=False):
        if with_target:
            return self.r.to_batch(kind, columns=[self.r.target, self.column])
        else:
            X, _ = self.r.to_batch(kind, columns=[self.column])
            return X
