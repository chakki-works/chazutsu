import os
from tqdm import tqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class CustomerReview(Dataset):

    def __init__(self, kind="products5"):
        super().__init__(
            name="Customer Review Data",
            site_url="https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets",  # noqa
            download_url="",
            description="customer review data from amazon.com that is annotated to each sentences"  # noqa
            )

        urls = {
            "products5": "http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip",  # noqa
            "additional9": "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/Reviews-9-products.zip",  # noqa
            "more3": "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/CustomerReviews-3-domains.zip",  # noqa
        }

        if kind not in urls:
            raise Exception("You have to choose kind from {}".format(
                ",".join(urls.keys()))
                )

        self.kind = kind
        self.download_url = urls[self.kind]

    @classmethod
    def products5(cls):
        return CustomerReview("products5")

    @classmethod
    def additional9(cls):
        return CustomerReview("additional9")

    @classmethod
    def more3(cls):
        return CustomerReview("more3")

    @property
    def root_name(self):
        return self.name.lower().replace(" ", "_") + "_" + self.kind

    def prepare(self, dataset_root, extracted_path):
        if self.kind == "products5":
            return self._prepare_products5(dataset_root, extracted_path)
        elif self.kind == "additional9":
            return self._prepare_additional9(dataset_root, extracted_path)
        elif self.kind == "more3":
            return self._prepare_more3(dataset_root, extracted_path)
        else:
            raise Exception("The kind {} is not supported.".format(self.kind))

        return extracted_path

    def make_resource(self, data_root):
        return Resource(
                data_root,
                columns=["sentence-type", "polarity", "detail", "review"],
                target="polarity")

    def _prepare_products5(self, dataset_root, extracted_path):
        products5_path = os.path.join(dataset_root, "products5.txt")
        source = os.path.join(extracted_path, "customer review data")
        with open(products5_path, mode="wb") as f:
            for txt in os.listdir(source):
                if txt == "Readme.txt":
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(source, txt)
                total = self.get_line_count(p)
                with open(p, encoding="utf-8") as rv:
                    for ln in tqdm(rv, total=total):
                        _ln = ln.strip().replace("\t", " ")
                        r = ReviewSentence.parse(_ln)
                        if r is not None:
                            if r.sentence_type:
                                f.write((r.to_row() + "\n").encode("utf-8"))
                            else:
                                skipped.append(_ln)

                if len(skipped) > 0:
                    self.logger.warning(
                        " {} lines is skipped "
                        "because annotation format is incorrect.".format(
                          len(skipped)))
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        return products5_path

    def _prepare_additional9(self, dataset_root, extracted_path):
        additional9_path = os.path.join(dataset_root, "additional9.txt")
        source = os.path.join(extracted_path, "Reviews-9-products")
        with open(additional9_path, mode="wb") as f:
            for txt in os.listdir(source):
                if txt == "Readme.txt":
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(source, txt)
                total = self.get_line_count(p)
                with open(p, encoding="utf-8", errors="replace") as rv:
                    for ln in tqdm(rv, total=total):
                        _ln = ln.strip().replace("\t", " ")
                        r = ReviewSentence.parse(_ln)
                        if r is not None:
                            if r.sentence_type:
                                f.write((r.to_row() + "\n").encode("utf-8"))
                            else:
                                skipped.append(_ln)

                if len(skipped) > 0:
                    self.logger.warning(
                        " {} lines is skipped because "
                        "annotation format is incorrect.".format(len(skipped)))
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        return additional9_path

    def _prepare_more3(self, dataset_root, extracted_path):
        more3_path = os.path.join(dataset_root, "more3.txt")
        source = os.path.join(extracted_path, "CustomerReviews-3domains(IJCAI2015)")
        with open(more3_path, mode="wb") as f:
            for txt in os.listdir(source):
                if txt == "Readme.txt":
                    continue

                if txt.endswith(".xml"):
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(source, txt)
                total = self.get_line_count(p)
                with open(p, encoding="utf-8") as rv:
                    for ln in tqdm(rv, total=total):
                        _ln = ln.strip().replace("\t", " ")
                        r = ReviewSentence.parse(_ln)
                        if r is not None:
                            if r.sentence_type:
                                f.write((r.to_row() + "\n").encode("utf-8"))
                            else:
                                skipped.append(_ln)

                if len(skipped) > 0:
                    self.logger.warning(
                        " {} lines is skipped because "
                        "annotation format is incorrect.".format(len(skipped)))
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        return more3_path


class ReviewSentence():

    def __init__(self, sentence_type, body, detail="", polarity=0):
        self.sentence_type = sentence_type
        self.detail = detail
        self.body = body
        self.polarity = polarity

    @classmethod
    def parse(cls, sentence):
        s = sentence.strip()
        if not s:
            return None

        if s.startswith("*"):
            return None  # comment

        if s.startswith("[t]"):
            return ReviewSentence("t", s[len("[t]"):])

        attr_body = s.split("##")
        if len(attr_body) != 2:
            attr_body = s.split("#")
            if len(attr_body) != 2:
                return ReviewSentence("", "")

        attr, body = attr_body
        if not attr:
            return ReviewSentence("-", body)

        attrs = attr.split(",")
        details = []
        scores = []
        for a in attrs:
            _a = a.strip()
            if not _a:
                continue
            detail = _a.replace("[", "_").replace("{", "_")
            detail = detail.replace("]", "").replace("}", "")
            ds = detail.split("_")

            if len(ds) == 2:  # no attribute
                detail += "_"
            elif len(ds) == 1:  # annotation miss
                break

            if len(ds) > 1:
                if ds[1] == "+":
                    scores.append(1)
                elif ds[1] == "-":
                    scores.append(-1)
                elif ds[1].replace("+", "").replace("-", "").isdigit():
                    scores.append(int(ds[1]))
                else:
                    continue
            else:
                continue
            details.append(detail)

        if len(scores) == 0:
            return ReviewSentence("", "")

        score = sum(scores) / len(scores)
        return ReviewSentence("po", body, ",".join(details), score)

    def to_row(self):
        return "\t".join([
            self.sentence_type,
            str(self.polarity),
            self.detail,
            self.body])
