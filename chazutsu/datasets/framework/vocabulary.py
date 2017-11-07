import os
import sys
import mmap
import numpy as np
from collections import Counter
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.tokenizer import Tokenizer


class Vocabulary():

    def __init__(self, root, name, tokenizer=None,
                 unknown="<unk>", padding="<pad>",
                 end_of_sentence=""):
        self.root = root
        self.name = name
        self._vocab_file_path = os.path.join(root, name + ".vocab")
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
        self.unknown = unknown
        self.padding = padding
        self.end_of_sentence = end_of_sentence
        self._vocab = {}
        self.__rev_vocab = {}
        self.max_len = 0

        # create logger
        from logging import getLogger, StreamHandler, DEBUG
        self.logger = getLogger(self.name.lower() + "_vocabulary")
        if not self.logger.hasHandlers():
            # logger is global object!
            _level = DEBUG
            handler = StreamHandler(sys.stdout)
            handler.setLevel(_level)
            self.logger.setLevel(_level)
            self.logger.addHandler(handler)

        if self.has_vocab():
            self.load()

    def has_vocab(self):
        return os.path.exists(self._vocab_file_path)

    def __len__(self):
        return len(self._vocab)

    def make(self,
             path_or_paths, vocab_size=-1, min_word_freq=0,
             separator="\t", reserved_words=(), target_column_indexes=()):
        vocab = Counter()
        paths = path_or_paths
        if isinstance(paths, str):
            paths = [paths]

        self.max_len = 0
        for p in paths:
            self.logger.info("Read {} to make vocabulary.".format(p))
            count = self.get_line_count(p)
            for words in xtqdm(self.fetch_line(p, target_column_indexes,
                               separator), total=count):
                for w in words:
                    vocab[w] += 1
                if len(words) > self.max_len:
                    self.max_len = len(words)

        _vocab = [k_v[0] for k_v in vocab.most_common()
                  if not k_v[1] < min_word_freq]
        _rv = reserved_words
        if len(_rv) == 0:
            _rv = [w for w in
                   [self.padding, self.unknown, self.end_of_sentence] if w]
        _vocab = list(_rv) + _vocab

        if vocab_size > 0:
            _vocab = _vocab[:vocab_size]

        self.logger.info(
            "The vocabulary count is {}. You can see it in {}.".format(
                len(_vocab), self._vocab_file_path))
        with open(self._vocab_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_vocab))
        self._vocab = dict(zip(_vocab, range(len(_vocab))))
        self.__rev_vocab = {}

    def get_line_count(self, file_path):
        count = 0
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    def fetch_line(self, path, target_column_indexes=(), separator="\t"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                _line = line
                if len(target_column_indexes) > 0:
                    els = _line.split(separator)
                    els = [els[i] for i in target_column_indexes]
                    _line = " ".join(els)

                words = self.tokenizer.tokenize(_line)
                if self.end_of_sentence:
                    words.append(self.end_of_sentence)

                yield words

    def load(self):
        if self.has_vocab():
            self.__rev_vocab = {}
            with open(self._vocab_file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for i, ln in enumerate(lines):
                    w = ln.strip()
                    self._vocab[w] = i

    def str_to_ids(self, sentence):
        if len(self._vocab) == 0:
            self.load()
        _s = sentence
        if self.end_of_sentence:
            _s = _s.replace("\r\n", self.end_of_sentence)
            _s = _s.replace("\n", self.end_of_sentence)
        words = self.tokenizer.tokenize(sentence)
        unk_id = self._vocab[self.unknown]
        ids = [unk_id if w not in self._vocab else self._vocab[w]
               for w in words]
        return ids

    def ids_to_words(self, ids, ignore_padding=False):
        if len(self._vocab) == 0:
            self.load()

        if len(self.__rev_vocab) == 0:
            self.__rev_vocab = {v: k for k, v in self._vocab.items()}

        words = [self.__rev_vocab[i] for i in ids]
        if ignore_padding:
            words = [w for w in words if w != self.padding]
        return words

    def str_to_matrix(self, sentence, fixed_len=-1):
        ids = self.str_to_ids(sentence)
        pad_id = self._vocab[self.padding]
        if fixed_len > 0:
            if len(ids) > fixed_len:
                ids = ids[:fixed_len]
            elif len(ids) < fixed_len:
                ids = ids + ([pad_id] * (fixed_len - len(ids)))

        seqlen_x_vocab = np.zeros((len(ids), len(self._vocab)))
        for i, _id in enumerate(ids):
            seqlen_x_vocab[i][_id] = 1
        return seqlen_x_vocab

    def matrix_to_words(self, matrix, ignore_padding=False):
        ids = np.argmax(matrix, axis=1)
        return self.ids_to_words(ids, ignore_padding)
