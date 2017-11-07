from collections import Counter
import numpy as np
import pandas as pd


class Converter():

    def __init__(self):
        pass

    def flow(self, series):
        return self.to_seq(series)

    def back(self, converted):
        return converted

    def to_seq(self, series):
        seq = series
        if isinstance(seq, pd.Series):
            seq = series.values
        if isinstance(seq, (list, tuple)):
            seq = np.array(seq)
        return seq


class DictionalyConverter(Converter):

    def __init__(self, labels=(), distincts=False):
        super().__init__()
        self.initialize(labels, distincts)

    def initialize(self, labels, distincts):
        self.labels = self.to_seq(labels)
        if not distincts:
            c = Counter(self.labels)
            self.labels = [k_v[0] for k_v in c.most_common()]

    def flow(self, series):
        if len(self.labels) == 0:
            self.initialize(series, False)
        x = [self.labels.index(d) for d in self.to_seq(series)]
        return np.array(x)

    def back(self, converted):
        x = [self.labels[d] for d in converted]
        return np.array(x)


class OneHotConverter(DictionalyConverter):

    def __init__(self, labels, distincts=False, num_class=-1):
        super().__init__(labels, distincts)
        self.num_class = num_class

    def flow(self, series):
        if len(self.labels) == 0:
            self.initialize(series, False)

        x = np.zeros((len(series), max(len(self.labels), self.num_class)))
        for i, d in enumerate(self.to_seq(series)):
            _id = self.labels.index(d)
            x[i][_id] = 1
        return x

    def back(self, converted):
        ids = np.argmax(converted, axis=1)
        x = [self.labels[d] for d in ids]
        return np.array(x)


class VocabularyConverter(Converter):

    def __init__(self, vocabulary, fixed_len=-1):
        self._vocab = vocabulary
        self.fixed_len = fixed_len if fixed_len > 0 else vocabulary.max_len

    def flow(self, series):
        x = [self._vocab.str_to_matrix(d, fixed_len=self.fixed_len)
             for d in self.to_seq(series)]
        return np.array(x)

    def back(self, converted):
        x = [self._vocab.matrix_to_words(d) for d in converted]
        return np.array(x)
