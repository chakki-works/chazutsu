import os
import sys
import re
import mmap
import random
import requests
import zipfile
import tarfile
import gzip
import shutil
import math
from urllib.parse import urlparse

from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.resource import Resource

class Dataset():
    """ Dataset Class Framework """

    def __init__(self,
                 name, site_url, download_url, description,
                 log_level=None, test_download_url=""):
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
        # When test_mode = True, use the test_url
        self.test_mode = False
        self.test_download_url = test_download_url
        if not self.test_download_url:
            self.test_download_url = download_url
        self.__trush = []

        # create logger
        from logging import getLogger, StreamHandler, DEBUG
        self.logger = getLogger(self.__class__.__name__.lower())
        if not self.logger.hasHandlers():
            # logger is global object!
            _level = DEBUG if log_level is None else log_level
            handler = StreamHandler(sys.stdout)
            handler.setLevel(_level)
            self.logger.setLevel(_level)
            self.logger.addHandler(handler)

    @property
    def root_name(self):
        return self.name.lower().replace(" ", "_")
    
    @property
    def extract_targets(self):
        return ()

    def on_test(self):
        self.test_mode = True
        return self

    def get_dataset_root(self, directory):
        return os.path.join(directory, self.root_name)

    def get_extracted_path(self, compressed_file):
        dataset_root, file_name = os.path.split(compressed_file)
        return os.path.join(dataset_root, "_extracted")

    def save_and_extract(self, directory="", force=False):
        _dir = self.check_directory(directory)
        dataset_root = self.get_dataset_root(_dir)
        if not os.path.isdir(dataset_root):
            os.mkdir(dataset_root)

        # download and save file
        save_file_path = self.save_dataset(dataset_root)

        # extract dataset file from saved file
        extracted_file_path = self.extract(save_file_path)

        self.trush(save_file_path)
        return dataset_root, extracted_file_path

    def download(self,
                 directory="", shuffle=True, test_size=0.3, sample_count=0,
                 force=False):
        _dir = self.check_directory(directory)
        dataset_root = self.get_dataset_root(_dir)
        save_file_path = os.path.join(dataset_root, self._get_file_name(None))
        if os.path.isdir(dataset_root) and not os.path.exists(save_file_path) \
           and (not force and not self.test_mode):
            r = self.make_resource(dataset_root)
            if len(r._resource) > 0:
                # data_root already exists and have contents
                self.logger.info(
                    "Read resource from the existed resource"
                    "(if you want to retry, set force=True).")
                return self.make_resource(dataset_root)

        dataset_root, extracted_path = self.save_and_extract(
            directory, force)

        prepared_file_path = self.prepare(dataset_root, extracted_path)

        if not self.test_mode:
            self.clear_trush()

        if shuffle:
            self.logger.info("Shuffle the extracted dataset.")
            lines = []
            with open(prepared_file_path, encoding="utf-8") as f:
                lines = f.readlines()
            random.shuffle(lines)
            with open(prepared_file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        # make sample file
        sample_file_path = self.make_samples(prepared_file_path,
                                             sample_count=sample_count,
                                             sample_directory=dataset_root)

        # split to train & test
        self.train_test_split(sample_file_path, test_size)

        if not self.test_mode:
            self.clear_trush()

        self.logger.info(
            "Done all process! Make below files at {}".format(dataset_root))
        for f in os.listdir(dataset_root):
            if f.startswith("."):
                continue
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

    def check_directory(self, directory):
        if os.path.isdir(directory):
            return directory
        elif len(directory) > 0:
            os.mkdir(directory)
            return directory
        else:
            current = os.getcwd()
            data_dir = os.path.join(current, "data")
            if not os.path.isdir(data_dir):
                self.logger.info(
                    "Make directory for download at {}.".format(current))
                os.mkdir(data_dir)
            return data_dir

    def save_dataset(self, dataset_root):
        save_file_path = os.path.join(dataset_root, self._get_file_name(None))
        if os.path.exists(save_file_path):
            self.logger.info("The dataset file already exists.")
            return save_file_path

        url = self.test_download_url if self.test_mode else self.download_url
        # download and save it as raw file
        self.logger.info(
            "Begin downloading the {} dataset from {}.".format(self.name, url))
        resp = requests.get(self.download_url, stream=True)
        if not resp.ok:
            raise Exception("Can not get dataset from {}.".format(url))

        # save content in response to file
        file_name = self._get_file_name(resp)
        _, ext = os.path.splitext(file_name)
        save_file_path = os.path.abspath(os.path.join(dataset_root, file_name))
        self.logger.info("The dataset is saved to {}".format(save_file_path))
        self.save_response_content(resp, save_file_path)
        
        return save_file_path

    def extract(self, compressed_file):
        target = self.get_extracted_path(compressed_file)
        if len(self.extract_targets) == 0:
            self.extractall(compressed_file)
        else:
            self.extract_file(compressed_file, self.extract_targets)

        return target

    def extractall(self, compressed_file):
        target = self.get_extracted_path(compressed_file)
        if os.path.exists(target):
            self.logger.info("The file already expanded.")
            return target
        else:
            os.mkdir(target)

        if zipfile.is_zipfile(compressed_file):
            with zipfile.ZipFile(compressed_file) as z:
                z.extractall(path=target)
        elif tarfile.is_tarfile(compressed_file):
            with tarfile.open(compressed_file) as t:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(t, path=target)
        elif compressed_file.endswith(".gz"):
            os.mkdir(target)
            with gzip.open(compressed_file, "rb") as g:
                file_name = os.path.basename(compressed_file)
                file_base, _ = os.path.splitext(file_name)
                p = os.path.join(target, file_base)
                with open(p, "wb") as f:
                    for ln in g:
                        f.write(ln)

        self.trush(compressed_file)

        return target

    def extract_file(self, compressed_file, relative_pathes):
        # unpack the archive file and extract directed path file
        base = self.get_extracted_path(compressed_file)
        if os.path.exists(base):
            self.logger.info("The file already expanded.")
            return base
        else:
            os.mkdir(base)
        is_path_lists = isinstance(relative_pathes, (tuple, list))
        target = relative_pathes if is_path_lists else [relative_pathes]

        extracteds = []
        if zipfile.is_zipfile(compressed_file):
            with zipfile.ZipFile(compressed_file) as z:
                for n in filter(lambda n: n in target, z.namelist()):
                    file_name = os.path.basename(n)
                    p = os.path.join(base, file_name)
                    with open(p, "wb") as f:
                        f.write(z.read(n))
                    extracteds.append(p)

        elif tarfile.is_tarfile(compressed_file):
            with tarfile.open(compressed_file) as t:
                for m in filter(lambda m: m in target, t.getnames()):
                    file_name = os.path.basename(m)
                    p = os.path.join(base, file_name)
                    with open(p, "wb") as f:
                        with t.extractfile(m) as tf:
                            for c in tf:
                                f.write(c)
                    extracteds.append(p)

        elif compressed_file.endswith(".gz"):
            with gzip.open(compressed_file, "rb") as g:
                p = os.path.join(base, relative_pathes[0])
                extracteds.append(p)
                with open(p, "wb") as f:
                    for ln in g:
                        f.write(ln)

        # remove downloaded raw file (zip/tar.gz etc)
        self.trush(compressed_file)

        return extracteds
    
    def prepare(self, dataset_root, extracted_path):
        return extracted_path

    def move_extracteds(self, dataset_root, extracted_path, filter=""):
        moveds = []
        for e in self.extract_targets:
            if filter and filter not in e:
                continue
            file_name = os.path.basename(e)
            s = os.path.join(extracted_path, file_name)
            t = os.path.join(dataset_root, file_name)
            shutil.move(s, t)
            moveds.append(t)
        return moveds

    def train_test_split(self, sample_file_path, test_size):
        if test_size < 0 or test_size > 1:
            self.logger.error(
                "test_size have to be between 0 ~ 1." \
                "if you don't want to split, please set 0.")
            return []
        elif test_size == 0 or test_size == 1:
            return []

        self.logger.info("Split to train & test file.")

        total_count = self.get_line_count(sample_file_path)
        test_count = int(round(total_count * test_size))
        test_targets = random.sample(range(total_count), test_count)
        
        base, ext = os.path.splitext(sample_file_path)
        train_test_path = [base + x + ext for x in ["_train", "_test"]]
        train_file = open(train_test_path[0], "wb")
        test_file = open(train_test_path[1], "wb")

        with open(sample_file_path, "rb") as f:
            i = 0
            for line in xtqdm(f, total=total_count):
                target = test_file if i in test_targets else train_file
                target.write(line)
                i += 1

        train_file.close()
        test_file.close()

        self.logger.info(
            "Train & Test file is {}({}rows) & {}({}rows, {:.2f}%).".format(
             os.path.basename(train_test_path[0]), total_count - test_count,
             os.path.basename(train_test_path[1]), test_count, 
             test_count / total_count * 100)
            )

        return train_test_path

    def make_samples(self, original_file_path, sample_count, sample_directory=None):

        original_dir, original_file = os.path.split(original_file_path)
        if sample_directory is None:
            sample_directory = original_dir

        original_filename, original_ext = os.path.splitext(original_file)
        sample_filename = original_filename + "_samples" + original_ext

        sample_path = os.path.join(sample_directory, sample_filename)
        samples_file = open(sample_path, "wb")

        total_count = self.get_line_count(original_file_path)
        if sample_count == 0:
            sample_count = total_count

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
        self.logger.info("Sample file is {} (picking {} samples).".format(
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

        if self.test_mode:
            file_name = "test_" + file_name

        return file_name



    def trush(self, path):
        self.__trush.append(path)

    def clear_trush(self):
        for t in self.__trush:
            if not os.path.exists(t):
                continue
            else:
                if os.path.isfile(t):
                    os.remove(t)
                if os.path.isdir(t):
                    shutil.rmtree(t)

    def show(self):
        print("About {}".format(self.name))
        print(self.description)
        print("see also: {}".format(self.site_url))

    @staticmethod
    def get_line_count(file_path):
        count = 0
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            while buf.readline():
                count += 1
        return count
    
    @staticmethod
    def save_response_content(response, save_file_path):

        total_size = int(response.headers.get("content-length", 0))
        with open(save_file_path, "wb") as f:
            chunk_size = 1024
            limit = math.ceil(total_size / chunk_size)
            for data in xtqdm(response.iter_content(chunk_size=chunk_size),
                              total=limit, unit="B", unit_scale=True):
                f.write(data)