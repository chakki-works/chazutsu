import math
import os
from joblib import Parallel, delayed

from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource
from chazutsu.datasets.framework.xtqdm import xtqdm


class IMDB(Dataset):

    def __init__(self):
        super().__init__(
            name="Large Movie Review Dataset(IMDB)",
            site_url="http://ai.stanford.edu/~amaas/data/sentiment/",
            download_url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",  # noqa
            description="Movie review data is constructed by 25,000 reviews " \
                        "that have positive/negative annotation"
            )

    def download(self,
                 directory="", shuffle=True, test_size=0, sample_count=0,
                 force=False):
        if test_size != 0:
            raise Exception("The dataset is already splitted to train & test.")

        return super().download(directory, shuffle, 0, sample_count, force)

    def prepare(self, dataset_root, extracted_path):
        extracted_dir = os.path.join(extracted_path, "aclImdb")
        data_dirs = ["train", "test"]
        pathes = []
        for d in data_dirs:
            target_dir = os.path.join(extracted_dir, d)
            file_path = os.path.join(dataset_root, "imdb_" + d + ".txt")
            self.label_by_dir(
                file_path, target_dir, {"pos": 1, "neg": 0}, task_size=1000)

            pathes.append(file_path)

            if d == "train":
                unlabeled = os.path.join(dataset_root, "imdb_unlabeled.txt")
                self.label_by_dir(
                    unlabeled, target_dir, {"unsup": None}, task_size=1000)
                pathes.append(unlabeled)

        return pathes[0]

    def make_resource(self, data_root):
        return IMDBResource(data_root)

    def label_by_dir(self, file_path, target_dir, dir_and_label, task_size=10):
        label_dirs = dir_and_label.keys()
        dirs = [d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d))
                and d in label_dirs]

        write_flg = True
        for d in dirs:
            self.logger.info(
                "Extracting {} (labeled by {}).".format(d, dir_and_label[d]))
            label = dir_and_label[d]
            dir_path = os.path.join(target_dir, d)
            pathes = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            pathes = [p for p in pathes if os.path.isfile(p)]
            task_length = int(math.ceil(len(pathes) / task_size))
            for i in xtqdm(range(task_length)):
                index = i * task_size
                tasks = pathes[index:(index + task_size)]
                lines = Parallel(n_jobs=-1)(
                    delayed(self._make_pair)(label, t) for t in tasks)
                mode = "w" if write_flg else "a"
                with open(file_path, mode=mode, encoding="utf-8") as f:
                    for ln in lines:
                        f.write(ln)
                write_flg = False

    @classmethod
    def _make_pair(cls, label, path):
        features = cls._file_to_features(path)
        line = "\t".join([str(label)] + features) + "\n"
        return line

    @classmethod
    def _file_to_features(cls, path):
        # override this method if you want implements custome process
        fs = []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [ln.replace("\t", " ").strip() for ln in lines]
            fs = [" ".join(lines)]
        return fs

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
                 columns=None, target="",
                 separator="\t", pattern=()):

        super().__init__(
            root,
            ["polarity", "rating", "review"],
            "polarity",
            separator,
            {
                "train": "_train",
                "test": "_test",
                "valid": "_valid",
                "unlabeled": "_unlabeled",
                "sample": "_samples"
            })

    @property
    def unlabeled_file_path(self):
        return self._get_prop("unlabeled")

    def unlabeled_data(self, split_target=False):
        return self._get_data("unlabeled", split_target)
