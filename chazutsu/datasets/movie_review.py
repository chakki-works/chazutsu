import os
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class MovieReview(Dataset):

    def __init__(self, kind="polarity"):
        super().__init__(
            name="Moview Review Data",
            site_url="http://www.cs.cornell.edu/people/pabo/movie-review-data/",  # noqa
            download_url="",
            description="movie review data is annotated by 3 kinds of label" \
            " (polarity, subjective rating, subjectivity)."
            )

        urls = {
            "polarity": "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz",  # noqa
            "polarity_v1": "http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",  # noqa
            "rating": "http://www.cs.cornell.edu/people/pabo/movie-review-data/scale_data.tar.gz",  # noqa
            "subjectivity": "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"  # noqa
        }

        if kind not in urls:
            keys = ",".join(urls.keys())
            raise Exception("You have to choose kind from {}".format(keys))

        self.kind = kind
        self.download_url = urls[self.kind]

    @classmethod
    def polarity(cls):
        return MovieReview("polarity")

    @classmethod
    def polarity_v1(cls):
        return MovieReview("polarity_v1")

    @classmethod
    def rating(cls):
        return MovieReview("rating")

    @classmethod
    def subjectivity(cls):
        return MovieReview("subjectivity")

    @property
    def root_name(self):
        return self.name.lower().replace(" ", "_") + "_" + self.kind

    @property
    def extract_targets(self):
        if self.kind == "polarity_v1":
            return ["rt-polaritydata/rt-polarity.neg",
                    "rt-polaritydata/rt-polarity.pos"]
        elif self.kind == "subjectivity":
            return ["plot.tok.gt9.5000",
                    "quote.tok.gt9.5000"]
        return ()

    def prepare(self, dataset_root, extracted_path):
        if self.kind == "polarity":
            return self._prepare_polarity(dataset_root, extracted_path)
        elif self.kind == "polarity_v1":
            return self._prepare_polarity_v1(dataset_root, extracted_path)
        elif self.kind == "rating":
            return self._prepare_rating(dataset_root, extracted_path)
        elif self.kind == "subjectivity":
            return self._prepare_subjectivity(dataset_root, extracted_path)
        else:
            raise Exception(
                "{} is not supported in extraction process.".format(self.kind))

    def make_resource(self, data_root):
        if self.kind in ["polarity", "polarity_v1"]:
            return Resource(data_root,
                            columns=["polarity", "review"], target="polarity")
        elif self.kind == "rating":
            return Resource(data_root,
                            columns=["rating", "review"], target="rating")
        elif self.kind == "subjectivity":
            return Resource(data_root,
                            columns=["subjectivity", "review"],
                            target="subjectivity")
        else:
            return Resource(data_root)

    def _prepare_polarity(self, dataset_root, extracted_path):
        polarity_file_path = os.path.join(dataset_root, "review_polarity.txt")
        negative_path = os.path.join(extracted_path, "txt_sentoken/neg")
        positive_path = os.path.join(extracted_path, "txt_sentoken/pos")

        with open(polarity_file_path, mode="w", encoding="utf-8") as f:
            for i, p in enumerate([negative_path, positive_path]):
                label = i  # negative = 0, positive = 1
                label_name = "negative" if label == 0 else "positive"
                self.logger.info("Extracting {} data.".format(label_name))
                for txt in xtqdm(os.listdir(p)):
                    with open(os.path.join(p, txt), encoding="utf-8") as tf:
                        lines = [ln.strip().replace("\t", " ") for ln in tf.readlines()]
                        review = " ".join(lines)
                        f.write("\t".join([str(label), review]) + "\n")

        return polarity_file_path

    def _prepare_polarity_v1(self, dataset_root, extracted_path):
        polarity_file = os.path.join(dataset_root, "review_polarity_v1.txt")
        with open(polarity_file, mode="w", encoding="utf-8") as f:
            for e in self.extract_targets:
                p = os.path.join(extracted_path, os.path.basename(e))
                label = 0 if e.endswith(".neg") else 1
                label_name = "negative" if label == 0 else "positive"
                self.logger.info("Extracting {} data.".format(label_name))
                total = self.get_line_count(p)
                with open(p, mode="r", errors="replace", encoding="utf-8") as p:
                    for ln in xtqdm(p, total=total):
                        review = ln.strip().replace("\t", " ")
                        f.write("\t".join([str(label), review]) + "\n")

        return polarity_file

    def _prepare_rating(self, dataset_root, extracted_path):
        rating_file_path = os.path.join(dataset_root, "review_rating.txt")
        rating_dir = os.path.join(extracted_path, "scaledata")

        rating_file = open(rating_file_path, "w", encoding="utf-8")
        for user in os.listdir(rating_dir):
            user_dir = os.path.join(rating_dir, user)
            if not os.path.isdir(user_dir):
                continue

            sub_in_review_file = os.path.join(user_dir, "subj." + user)
            user_rating_file = os.path.join(user_dir, "rating." + user)
            total = self.get_line_count(sub_in_review_file)
            self.logger.info("Extracting user {}'s rating data.".format(user))
            with open(sub_in_review_file, "r", encoding="utf-8") as sr:
                with open(user_rating_file, "r", encoding="utf-8") as ur:
                    for review, rating in xtqdm(zip(sr, ur), total=total):
                        _rv = review.strip().replace("\t", " ")
                        _r = rating.strip()
                        rating_file.write("\t".join([_r, _rv]) + "\n")

        rating_file.close()

        return rating_file_path

    def _prepare_subjectivity(self, dataset_root, extracted_path):
        subjectivity_file = os.path.join(dataset_root, "subjectivity.txt")
        with open(subjectivity_file, mode="w", encoding="utf-8") as f:
            for e in self.extract_targets:
                # subjective(plot) = 1
                label = 1 if e.startswith("plot.") else 0
                label_name = "subjective" if label == 1 else "objective"
                self.logger.info("Extracting {} data.".format(label_name))
                p = os.path.join(extracted_path, os.path.basename(e))
                total = self.get_line_count(p)
                with open(p, mode="r", errors="replace", encoding="utf-8") as sb:
                    for ln in xtqdm(sb, total=total):
                        review = ln.strip().replace("\t", " ")
                        f.write("\t".join([str(label), review]) + "\n")

        return subjectivity_file
