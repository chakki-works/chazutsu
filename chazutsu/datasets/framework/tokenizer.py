import os
import re


class Tokenizer():

    def __init__(self):
        self._line_strip_pattern = re.compile("(\.|\!|\?|\"|\'|,|\(|\)|\<|\>)")

    def tokenize(self, sentence):
        stripped = self.line_strip(sentence)
        words = self._tokenize(stripped)
        words = [w.strip() for w in words]
        words = [w for w in words if w]
        return words

    def line_strip(self, sentence):
        return re.sub(self._line_strip_pattern, " ", sentence)

    def _tokenize(self, stripped_sentence):
        # You have to implement how to tokenize the sentence
        words = stripped_sentence.split(" ")
        return words
