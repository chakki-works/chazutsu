import os
import zipfile
import shutil
from tqdm import tqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class CustomerReview(Dataset):

    def __init__(self, kind="products5"):
        super().__init__(
            name="Customer Review Data",
            site_url="https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets",
            download_url="",
            description="customer review data from amazon.com that is annotated to each sentences"
            )
        
        urls = {
            "products5": "http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip",
            "additional9": "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/Reviews-9-products.zip",
            "more3": "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/CustomerReviews-3-domains.zip",
        }

        if kind not in urls:
            raise Exception("You have to choose kind from {}".format(",".join(urls.keys())))

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

    def extract(self, path):
        if self.kind == "products5":
            return self._extract_products5(path)
        elif self.kind == "additional9":
            return self._extract_additional9(path)
        elif self.kind == "more3":
            return self._extract_more3(path)
        else:
            raise Exception("Directed kind {} is not supported in extraction process.".format(self.kind))

    def make_resource(self, data_root):
        return Resource(data_root, columns=["sentence-type", "polarity", "detail", "review"], target="polarity")

    def _extract_products5(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        products5_path = os.path.join(dir, "products5.txt")

        with zipfile.ZipFile(path) as z:
            z.extractall(path=work_dir)

        review_path = os.path.join(work_dir, "customer review data")
        with open(products5_path, mode="wb") as f:
            for txt in os.listdir(review_path):
                if txt == "Readme.txt":
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(review_path, txt)
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
                        " {} lines is skipped because of annotation format is not correct.".format(len(skipped))
                        )
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        # remove files
        os.remove(path)
        shutil.rmtree(work_dir)

        return products5_path

    def _extract_additional9(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        additional9_path = os.path.join(dir, "additional9.txt")

        with zipfile.ZipFile(path) as z:
            z.extractall(path=work_dir)

        review_path = os.path.join(work_dir, "Reviews-9-products")
        with open(additional9_path, mode="wb") as f:
            for txt in os.listdir(review_path):
                if txt == "Readme.txt":
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(review_path, txt)
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
                        " {} lines is skipped because of annotation format is not correct.".format(len(skipped))
                        )
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        # remove files
        os.remove(path)
        shutil.rmtree(work_dir)

        return additional9_path

    def _extract_more3(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        more3_path = os.path.join(dir, "more3.txt")

        with zipfile.ZipFile(path) as z:
            z.extractall(path=work_dir)

        review_path = os.path.join(work_dir, "CustomerReviews-3domains(IJCAI2015)")
        with open(more3_path, mode="wb") as f:
            for txt in os.listdir(review_path):
                if txt == "Readme.txt":
                    continue

                if txt.endswith(".xml"):
                    continue

                self.logger.info("Extracting {} data.".format(txt))
                skipped = []
                p = os.path.join(review_path, txt)
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
                        " {} lines is skipped because of annotation format is not correct.".format(len(skipped))
                        )
                    self.logger.debug(
                        "\n".join(map(lambda s: " >>{}".format(s), skipped))
                        )

        # remove files
        os.remove(path)
        shutil.rmtree(work_dir)

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
            detail = _a.replace("[", "_").replace("{", "_").replace("]", "").replace("}", "")
            ds = detail.split("_")

            if len(ds) == 2:  # no attribute
                detail += "_"
            elif len(ds) == 1: # annotation miss
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
    