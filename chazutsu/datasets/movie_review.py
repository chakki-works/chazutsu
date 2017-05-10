import os
import tarfile
import shutil
from tqdm import tqdm
from chazutsu.datasets.framework.dataset import Dataset


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

    def _extract_polarity(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        polarity_file_path = os.path.join(dir, "review_polarity.txt")

        with tarfile.open(path) as t:
            t.extractall(path=work_dir)

        negative_path = os.path.join(work_dir, "txt_sentoken/neg")
        positive_path = os.path.join(work_dir, "txt_sentoken/pos")

        with open(polarity_file_path, mode="w") as f:
            for i, p in enumerate([negative_path, positive_path]):
                label = i # negative = 0, positive = 1
                self.logger.info(
                    "Extract data from polarity file ({}).".format("negative" if label == 0 else "positive")
                    )
                for txt in tqdm(os.listdir(p)):
                    with open(os.path.join(p, txt)) as tf:
                        lines = [ln.strip().replace("\t", " ") for ln in tf.readlines()]
                        review= " ".join(lines)
                        f.write("\t".join([str(label), review]) + os.linesep)

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
        with open(polarity_file, mode="w") as f:
            for e in extracteds:
                label = 0 if e.endswith(".neg") else 1
                self.logger.info(
                    "Extract data from polarity_v1 file ({}).".format("negative" if label == 0 else "positive")
                    )
                total = self.get_line_count(e)
                with open(e, mode="r", errors="replace") as p:
                    for ln in tqdm(p, total=total):
                        review = ln.strip().replace("\t", " ")
                        f.write("\t".join([str(label), review]) + os.linesep)
        
        for e in extracteds:
            os.remove(e)
        
        return polarity_file
