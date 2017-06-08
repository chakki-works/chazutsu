import os
import sys
import mmap
from collections import Counter
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.tokenizer import Tokenizer


class Vocabulary():

    def __init__(self, root, name, tokenizer=None, end_of_sentence="", unknown="<unk>"):
        self.root = root
        self.name = name
        self._vocab_file_path = os.path.join(root, name + ".vocab")
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
        self.end_of_sentence = end_of_sentence
        self.unknown = unknown
        self._vocab = {}
        self.__rev_vocab = {}

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

    def make(self, path_or_paths, vocab_size=-1, min_word_count=0, target_column_indexes=(), separator="\t"):
        vocab = Counter()
        paths = path_or_paths
        if isinstance(paths, str):
            paths = [paths]

        for p in paths:
            self.logger.info("Read {} to make vocabulary.".format(p))
            count = self.get_line_count(p)
            for words in xtqdm(self.fetch_line(p, target_column_indexes, separator), total=count):
                for w in words:
                    vocab[w] += 1
        
        _vocab = [k_v[0] for k_v in vocab.most_common() if not k_v[1] < min_word_count]
        if vocab_size > 0:
            _vocab = _vocab[:vocab_size]

        if self.unknown and self.unknown not in _vocab:
            _vocab.append(self.unknown)
        if self.end_of_sentence and self.end_of_sentence not in _vocab:
            _vocab.append(self.end_of_sentence)

        self.logger.info("The vocabulary count is {}. You can see it in {}.".format(
            len(_vocab), self._vocab_file_path
            ))
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
        words = self.tokenizer.tokenize(sentence)
        unk_id = self._vocab[self.unknown]
        ids = [unk_id if w not in self._vocab else self._vocab[w] for w in words]
        if self.end_of_sentence and sentence.endswith(os.linesep):
            eos_id = self._vocab[self.end_of_sentence]
            ids.append(eos_id)

        return ids

    def ids_to_words(self, ids):
        if len(self._vocab) == 0:
            self.load()
        
        if len(self.__rev_vocab) == 0:
            self.__rev_vocab = {v:k for k, v in self._vocab.items()}

        words = [self.__rev_vocab[i] for i in ids]
        return words
