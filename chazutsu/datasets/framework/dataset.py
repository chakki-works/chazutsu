import os
import sys
import re
import mmap
import random
import math
import zipfile
import tarfile
import gzip
from urllib.parse import urlparse
import requests
from chazutsu.datasets.framework.xtqdm import xtqdm
from joblib import Parallel, delayed
from chazutsu.datasets.framework.resource import Resource


class Dataset():
    """ Dataset Class Framework """

    def __init__(self, name, site_url, download_url, description, log_level=None):
        """
        Args:
            name        : dataset name
            site_url    : url to access the dataset homepage
            download_url    : dataset download url
            description     : description of dataset
        """

        self.name = name
        self.site_url = site_url
        self.download_url = download_url
        self.description = description

        # create logger
        from logging import getLogger, StreamHandler, DEBUG
        self.logger = getLogger("_".join(["dataset", self.__class__.__name__.lower()]))
        if not self.logger.hasHandlers():
            # logger is global object!
            _level = DEBUG if log_level is None else log_level
            handler = StreamHandler(sys.stdout)
            handler.setLevel(_level)
            self.logger.setLevel(_level)
            self.logger.addHandler(handler)
    
    def download(self, directory="", shuffle=True, test_size=0.3, sample_count=0, keep_raw=False, force=False):
        # input parameter check
        #   directory: don't make default directory in download method because can not specify current dir.
        #   test_size: have to be smaller than 1
        dir = self.check_directory(directory)
        dataset_root = os.path.join(dir, self._get_root_name())
        if not os.path.isdir(dataset_root):
            os.mkdir(dataset_root)
        elif not force:
            # data_root already exists and have contents
            self.logger.info("Read resource from the existed resource (if you want to retry, set force=True).")
            return self.make_resource(dataset_root)

        # download and save file
        save_file_path = self.save_dataset(dataset_root)

        # extract dataset file from saved file
        extracted_file_path = self.extract(save_file_path)

        if shuffle:
            self.logger.info("Shuffle the extracted dataset.")
            lines = []
            with open(extracted_file_path, encoding="utf-8") as f:
                lines = f.readlines()
            random.shuffle(lines)
            with open(extracted_file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        # make sample file
        sample_path = self.make_samples(extracted_file_path, sample_count)

        # split to train & test
        train_test_path = self.train_test_split(extracted_file_path, test_size, keep_raw)
        
        self.logger.info("Done all process! Make below files at {}".format(dataset_root))
        for f in os.listdir(dataset_root):
            self.logger.info(" " + f)
        
        r = self.make_resource(dataset_root)

        return r
    
    def load(self, directory=""):
        dir = self.check_directory(directory)
        dataset_root = os.path.join(dir, self._get_root_name())
        if os.path.isdir(dataset_root):
            return self.make_resource(dataset_root)
        else:
            return None
    
    def _get_root_name(self):
        return self.name.lower().replace(" ", "_")

    def check_directory(self, directory):
        if os.path.isdir(directory):
            return directory
        else:
            current = os.getcwd()
            self.logger.info("Make directory for downloading the file to {}.".format(current))
            data_dir = os.path.join(current, "data")
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            return data_dir

    def save_dataset(self, dataset_root):
        save_file_path = os.path.join(dataset_root, self._get_file_name(None))
        if os.path.exists(save_file_path):
            self.logger.info("Skip the downloading the file because it already exists.")
            return save_file_path

        # download and save it as raw file
        self.logger.info("Begin downloading the {} dataset from {}.".format(self.name, self.download_url))
        resp = requests.get(self.download_url, stream=True)
        if not resp.ok:
            raise Exception("Can not get dataset from {}.".format(self.download_url))
        
        # save content in response to file
        total_size = int(resp.headers.get("content-length", 0))
        file_name = self._get_file_name(resp)
        _, ext = os.path.splitext(file_name)
        save_file_path = os.path.abspath(os.path.join(dataset_root, file_name))
        self.logger.info("The dataset file is saved to {}".format(save_file_path))
        with open(save_file_path, "wb") as f:
            chunk_size = 1024
            limit = total_size / chunk_size
            for data in xtqdm(resp.iter_content(chunk_size=chunk_size), total=limit, unit="B", unit_scale=True):
                f.write(data)
        
        return save_file_path
    
    def extract(self, path):
        # you may unpack the file and extract data file.
        return path
    
    def extract_file(self, path, relative_pathes, remove=True):
        # unpack the archive file and extract directed path file
        base, _ = os.path.splitext(path)
        target = relative_pathes if isinstance(relative_pathes, (tuple, list)) else [relative_pathes]

        extracteds = []
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for n in filter(lambda n: n in target, z.namelist()):
                    file_name = os.path.basename(n)
                    p = os.path.join(os.path.dirname(path), file_name)
                    with open(p, "wb") as f:
                        f.write(z.read(n))
                    extracteds.append(p)

        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as t:
                for m in filter(lambda m: m in target, t.getnames()):
                    file_name = os.path.basename(m)
                    p = os.path.join(os.path.dirname(path), file_name)
                    with open(p, "wb") as f:
                        with t.extractfile(m) as tf:
                            for c in tf:
                                f.write(c)
                    extracteds.append(p)
        
        elif path.endswith(".gz"):
            with gzip.open(path, "rb") as g:
                p = os.path.join(os.path.dirname(path), relative_pathes[0])
                extracteds.append(p)
                with open(p, "wb") as f:
                    for ln in g:
                        f.write(ln)

        if remove:
            # remove downloaded raw file (zip/tar.gz etc)
            os.remove(path)

        return extracteds
    
    def label_by_dir(self, file_path, target_dir, dir_and_label, task_size=10):

        label_dirs = dir_and_label.keys()
        dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and d in label_dirs]

        write_flg = True
        for d in dirs:
            self.logger.info("Extracting {} directory files (labeled by {}).".format(d, dir_and_label[d]))
            label = dir_and_label[d]
            dir_path = os.path.join(target_dir, d)
            pathes = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            pathes = [p for p in pathes if os.path.isfile(p)]
            task_length = int(math.ceil(len(pathes) / task_size))
            for i in xtqdm(range(task_length)):
                index = i * task_size
                task_packet = pathes[index:(index + task_size)]
                lines = Parallel(n_jobs=-1)(delayed(self._parallel_parser)(label, t) for t in task_packet)
                with open(file_path, mode="w" if write_flg else "a", encoding="utf-8") as f:
                    for ln in lines:
                        f.write(ln)
                write_flg = False
    
    @classmethod
    def _parallel_parser(cls, label, path):
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

    def train_test_split(self, original_file_path, test_size, keep_raw=False):
        if test_size < 0 or test_size > 1:
            self.logger.error("test_size have to be between 0 ~ 1. if you don't want to split, please set 0.")
            return []
        elif test_size == 0 or test_size == 1:
            return []

        self.logger.info("Split to train & test file.")

        total_count = self.get_line_count(original_file_path)
        test_count = int(round(total_count * test_size))
        test_targets = random.sample(range(total_count), test_count)
        
        base, ext = os.path.splitext(original_file_path)
        train_test_path = [base + x + ext for x in ["_train", "_test"]]
        train_file = open(train_test_path[0], "wb")
        test_file = open(train_test_path[1], "wb")

        with open(original_file_path, "rb") as f:
            i = 0
            for line in xtqdm(f, total=total_count):
                target = test_file if i in test_targets else train_file
                target.write(line)
                i += 1

        train_file.close()
        test_file.close()

        to_base = lambda x: os.path.basename(x)
        self.logger.info("File is splited to {} & {}. Each records are {} & {} (test_size={:.2f}%).".format(
            to_base(train_test_path[0]), to_base(train_test_path[1]),
            total_count - test_count, test_count, test_count / total_count * 100)
            )

        if not keep_raw:
            os.remove(original_file_path)

        return train_test_path

    def make_samples(self, original_file_path, sample_count):
        if sample_count == 0:
            return ""
        
        base, ext = os.path.splitext(original_file_path)
        sample_path = base + "_samples" + ext
        samples_file = open(sample_path, "wb")

        total_count = self.get_line_count(original_file_path)
        # for reproducibility of sampling, use fixed interval
        sample_target = range(0, total_count, total_count // sample_count)
        with open(original_file_path, "rb") as f:
            count = 0
            i = 0
            for line in f:
                if i in sample_target:
                    samples_file.write(line)
                    count += 1
                i += 1
                if count == sample_count:
                    break

        samples_file.close()
        self.logger.info("Make {} by picking {} records from original file.".format(
            os.path.basename(sample_path), sample_count)
        )

        return sample_path
    
    def make_resource(self, data_root):
        return Resource(data_root)

    def _get_file_name(self, resp):
        file_name = ""
        if resp and "content-disposition" in resp.headers:
            cd = resp.headers["content-disposition"]
            file_matches = re.search("filename=(.+)", cd)
            if file_matches:
                file_name = file_matches.group(0)
                file_name = file_name.split("=")[1]
        else:
            parsed = urlparse(self.download_url)
            file_name = os.path.basename(parsed.path)

        return file_name

    def get_line_count(self, file_path):  
        count = 0
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    def show(self):
        print("About {}".format(self.name))
        print(self.description)
        print("see also: {}".format(self.site_url))

