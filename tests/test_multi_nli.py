import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase
from chazutsu.datasets.multi_nli import MultiNLI


class TestMultiNLI(DatasetTestCase):

    @classmethod
    def setUpClass(cls):
        DatasetTestCase.setUpClass()
        MultiNLI().save_and_extract(DatasetTestCase.class_test_dir)

    def test_preprocess(self):
        multi_nli = MultiNLI()
        values = multi_nli.preprocess_jsonl(test_jsonl)

        self.assertEqual(len(values), len(multi_nli.columns))
        print(values)

    def test_prepare(self):
        multi_nli = MultiNLI()
        dataset_root, extracted = multi_nli.save_and_extract(DatasetTestCase.class_test_dir)
        train_file = multi_nli.prepare(dataset_root, extracted)
        self.assertTrue(train_file)
        multi_nli.clear_trush()

    def test_tokenize(self):
        r = chazutsu.datasets.MultiNLI.matched().download(DatasetTestCase.class_test_dir)
        r.make_vocab(min_word_freq=5)


test_jsonl = """
{"annotator_labels": ["neutral"], "genre": "government", "gold_label": "neutral", "pairID": "335730n", "promptID": "335730n", "sentence1": "Conceptually cream skimming has two basic dimensions - product and geography.", "sentence1_binary_parse": "( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))", "sentence2": "Product and geography are what make cream skimming work. ", "sentence2_binary_parse": "( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))"}
"""

if __name__ == "__main__":
    unittest.main()
