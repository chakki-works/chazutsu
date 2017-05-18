import os
import tarfile
import shutil
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class MovieReview(Dataset):

    def __init__(self, kind="polarity"):
        super().__init__(
            name="Moview Review Data",
            site_url="http://www.cs.cornell.edu/people/pabo/movie-review-data/",
            download_url="",
            description="movie review data that is annotated by 3 kinds of label (polarity, subjective rating, subjectivity)."
            )
        
        urls = {
            "polarity": "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz",
            "polarity_v1": "http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
            "rating": "http://www.cs.cornell.edu/people/pabo/movie-review-data/scale_data.tar.gz",
            "subjectivity": "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"
        }

        if kind not in urls:
            raise Exception("You have to choose kind from {}".format(",".join(urls.keys())))

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

    def extract(self, path):
        if self.kind == "polarity":
            return self._extract_polarity(path)
        elif self.kind == "polarity_v1":
            return self._extract_polarity_v1(path)
        elif self.kind == "rating":
            return self._extract_rating(path)
        elif self.kind == "subjectivity":
            return self._extract_subjectivity(path)
        else:
            raise Exception("Directed kind {} is not supported in extraction process.".format(self.kind))

    def make_resource(self, data_root):
        if self.kind in ["polarity", "polarity_v1"]:
            return Resource(data_root, columns=["polarity", "review"], target="polarity")
        elif self.kind == "rating":
            return Resource(data_root, columns=["rating", "review"], target="rating")
        elif self.kind == "subjectivity":
            return Resource(data_root, columns=["subjectivity", "review"], target="subjectivity")
        else:
            return Resource(data_root)

    def _extract_polarity(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        polarity_file_path = os.path.join(dir, "review_polarity.txt")

        with tarfile.open(path) as t:
            t.extractall(path=work_dir)

        negative_path = os.path.join(work_dir, "txt_sentoken/neg")
        positive_path = os.path.join(work_dir, "txt_sentoken/pos")

        with open(polarity_file_path, mode="w", encoding="utf-8") as f:
            for i, p in enumerate([negative_path, positive_path]):
                label = i # negative = 0, positive = 1
                self.logger.info(
                    "Extracting {} data.".format("negative" if label == 0 else "positive")
                    )
                for txt in xtqdm(os.listdir(p)):
                    with open(os.path.join(p, txt), encoding="utf-8") as tf:
                        lines = [ln.strip().replace("\t", " ") for ln in tf.readlines()]
                        review= " ".join(lines)
                        f.write("\t".join([str(label), review]) + "\n")

        # remove files
        os.remove(path)
        shutil.rmtree(work_dir)

        return polarity_file_path

    def _extract_polarity_v1(self, path):
        dir, file_name = os.path.split(path)
        extracteds = self.extract_file(
            path, 
            ["rt-polaritydata/rt-polarity.neg","rt-polaritydata/rt-polarity.pos"],
            remove=True
        )

        polarity_file = os.path.join(dir, "review_polarity_v1.txt")
        with open(polarity_file, mode="w", encoding="utf-8") as f:
            for e in extracteds:
                label = 0 if e.endswith(".neg") else 1
                self.logger.info(
                    "Extracting {} data.".format("negative" if label == 0 else "positive")
                    )
                total = self.get_line_count(e)
                with open(e, mode="r", errors="replace", encoding="utf-8") as p:
                    for ln in xtqdm(p, total=total):
                        review = ln.strip().replace("\t", " ")
                        f.write("\t".join([str(label), review]) + "\n")
        
        for e in extracteds:
            os.remove(e)
        
        return polarity_file

    def _extract_rating(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        rating_file_path = os.path.join(dir, "review_rating.txt")

        with tarfile.open(path) as t:
            t.extractall(path=work_dir)
        
        rating_dir = os.path.join(work_dir, "scaledata")

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
        os.remove(path)
        shutil.rmtree(work_dir)

        return rating_file_path

    def _extract_subjectivity(self, path):
        dir, file_name = os.path.split(path)
        extracteds = self.extract_file(
            path, 
            ["plot.tok.gt9.5000","quote.tok.gt9.5000"],
            remove=True
        )

        subjectivity_file = os.path.join(dir, "subjectivity.txt")
        with open(subjectivity_file, mode="w", encoding="utf-8") as f:
            for e in extracteds:
                fname = os.path.basename(e)
                label = 1 if fname.startswith("plot.") else 0  # subjective(plot) = 1
                self.logger.info(
                    "Extracting {} data.".format("subjective" if label == 1 else "objective")
                    )
                total = self.get_line_count(e)
                with open(e, mode="r", errors="replace", encoding="utf-8") as sb:
                    for ln in xtqdm(sb, total=total):
                        review = ln.strip().replace("\t", " ")
                        f.write("\t".join([str(label), review]) + "\n")
        
        for e in extracteds:
            os.remove(e)
        
        return subjectivity_file
