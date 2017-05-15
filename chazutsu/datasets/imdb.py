import os
import shutil
import tarfile
import pandas as pd
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class IMDB(Dataset):

    def __init__(self):
        super().__init__(
            name="Large Movie Review Dataset(IMDB)",
            site_url="http://ai.stanford.edu/~amaas/data/sentiment/",
            download_url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            description="Movie review data that is constructed by 25,000 train/test reviews that have positive/negative annotation"
            )
    
    def download(self, directory="", shuffle=True, test_size=0, sample_count=0, keep_raw=False):
        if test_size != 0:
            raise Exception("This dataset is already splitted to train & test.")
        
        return super().download(directory, shuffle, 0, sample_count, keep_raw)
    
    def extract(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        if not os.path.isdir(work_dir):
            with tarfile.open(path) as t:
                t.extractall(path=work_dir)
        
        extracted_dir = os.path.join(work_dir, "aclImdb")
        data_dirs = ["train", "test"]
        pathes = []
        for d in data_dirs:
            target_dir = os.path.join(extracted_dir, d)
            file_path = os.path.join(dir, "imdb_" + d + ".txt")
            self.label_by_dir(file_path, target_dir, {"pos": 1, "neg": 0}, task_size=1000)

            pathes.append(file_path)

            if d == "train":
                unlabeled = os.path.join(dir, "imdb_unlabeled.txt")
                self.label_by_dir(unlabeled, target_dir, {"unsup": None}, task_size=1000)
                pathes.append(unlabeled)

        os.remove(path)
        shutil.rmtree(work_dir)

        return pathes[0]

    def make_resource(self, data_root):
        return IMDBResource(data_root)

    @classmethod
    def _parallel_parser(cls, label, path):
        features = cls._file_to_features(path)
        if label is not None:
            line = "\t".join([str(label)] + features) + "\n"
        else:
            line = "\t".join(features) + "\n"  # unlabeled
        return line

    @classmethod
    def _file_to_features(cls, path):
        # override this method if you want implements custome process
        fs = []
        file_name = os.path.basename(path)
        f, ext = os.path.splitext(file_name)
        els = f.split("_")
        rating = 0
        if len(els) == 2:
            rating = els[-1]
        
        review = ""
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [ln.replace("\t", " ").strip() for ln in lines]
            review = " ".join(lines)
        
        if rating != "0":
            return [rating, review]
        else:
            return [review]


class IMDBResource(Resource):

    def __init__(self, 
        root,
        train_file_suffix="_train",
        test_file_suffix="_test",
        sample_file_suffix="_samples"):

        super().__init__(
            root, 
            ["polarity", "rating", "review"],
            "polarity",
            train_file_suffix, 
            test_file_suffix, 
            sample_file_suffix)
        
        self.path = self.train_file_path
        self.unlabeled_data_path = ""
        for f in os.listdir(self.root):
            p = os.path.join(self.root, f)
            n, e = os.path.splitext(f)
            if n.endswith("_unlabeled"):
                self.unlabeled_data_path = p
                break
    
    def unlabeled_data(self):
        df = pd.read_table(self.unlabeled_data_path, header=None, names=["review"])
        return df
