import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import chazutsu.datasets
from tests.dataset_base_test import DatasetTestCase
from chazutsu.datasets.customer_review import ReviewSentence


class TestCustomerReview(DatasetTestCase):

    def test_review_sentence(self):
        s1 = "[t]excellent picture quality / color"
        s2 = "use[+2]##the camera is very easy to use , in fact on a recent"
        s3 = "camera[+2], use[+4]##bottom line , well made camera ,"
        s4 = "size[-2][u],weight[-2][u]##1 ) quite bulky ( it '"

        for i, s in enumerate([s1, s2, s3, s4]):
            rs = ReviewSentence.parse(s)
            print(rs.to_row())
            if i == 0:
                self.assertEqual("t", rs.sentence_type)
            elif i == 1:
                self.assertEqual(2, rs.polarity)
                self.assertEqual("use_+2_", rs.detail)
            elif i == 2:
                self.assertEqual(2, len(rs.detail.split(",")))
                self.assertEqual(3, rs.polarity)
            elif i == 3:
                self.assertEqual(2, len(rs.detail.split(",")))
                self.assertEqual(-2, rs.polarity)
                self.assertEqual("size_-2_u", rs.detail.split(",")[0])

    def test_prepare_products5(self):
        d = chazutsu.datasets.CustomerReview.products5()
        root, extracted = d.save_and_extract(self.test_dir)
        path = d._prepare_products5(root, extracted)

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.split("\t")
                    if len(els) != 4:
                        raise Exception("number of elements is not correct.")
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)
        d.trush(path)
        d.clear_trush()

    def test_prepare_additional9(self):
        d = chazutsu.datasets.CustomerReview.additional9()
        root, extracted = d.save_and_extract(self.test_dir)
        path = d._prepare_additional9(root, extracted)

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.split("\t")
                    if len(els) != 4:
                        raise Exception("number of elements is not correct.")
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)

        d.trush(path)
        d.clear_trush()

    def test_prepare_more3(self):
        d = chazutsu.datasets.CustomerReview.more3()
        root, extracted = d.save_and_extract(self.test_dir)
        path = d._prepare_more3(root, extracted)

        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    els = ln.split("\t")
                    if len(els) != 4:
                        raise Exception("number of elements is not correct.")
        except Exception as ex:
            d.clear_trush()
            self.fail(ex)

        d.trush(path)
        d.clear_trush()

    def test_download(self):
        resource = chazutsu.datasets.CustomerReview.more3().download(self.test_dir)
        self.assertTrue(len(resource.data().columns), 4)


if __name__ == "__main__":
    unittest.main()
