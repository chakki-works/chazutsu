import os
import tarfile
import shutil
from collections import namedtuple
import requests
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class ReutersNews(Dataset):

    def __init__(self, kind="topics"):
        super().__init__(
            name="ReutersNews",
            site_url="http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm",
            download_url="https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/RCV1-v2.zip",
            description="Reuters news corpus that is annotated by 3 kinds of label (topics, industries, region)."
            )
        
        label_urls = {
            "topics": "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz",
            "industries": "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a09-industry-qrels/rcv1-v2.industries.qrels.gz",
            "regions": "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a10-region-qrels/rcv1-v2.regions.qrels.gz",
        }

        if kind not in label_urls:
            raise Exception("You have to choose kind from {}".format(",".join(label_urls.keys())))

        self.kind = kind
        self.label_url = label_urls[self.kind]
        self.label_file = "rcv1-v2.{}.qrels".format(self.kind)
        self.label_desc_file = "rcv1.{}.txt".format(self.kind)
    
    def download(self, directory="", shuffle=False, test_size=0, sample_count=0, keep_raw=False, force=False):
        if test_size != 0:
            raise Exception("This dataset is already splitted to train & test.")
        
        return super().download(directory, False, 0, sample_count, keep_raw, force)

    @classmethod
    def topics(cls):
        return ReutersNews("topics")

    @classmethod
    def industries(cls):
        return ReutersNews("industries")

    @classmethod
    def regions(cls):
        return ReutersNews("regions")

    def read_label_desc(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # skip header
                ln = line.strip().split("\t")
                


    def extract(self, path):
        dir, fname = os.path.split(path)
        extracteds = self.extract_file(
            path, 
            [
                "RCV1-v2/lyrl2004_tokens_train.csv", 
                "RCV1-v2/lyrl2004_tokens_test.csv", 
                "RCV1-v2/" + self.label_desc_file
                ],
            remove=True
        )
        
        label_file_path = os.path.join(dir, self.label_file)
        if not os.path.exists(label_file_path):
            self.logger.info("Downloading the annotation file")
            dl_file_path = label_file_path + ".gz"
            self._download_file(self.label_url, dl_file_path)
            label_file_path = self.extract_file(dl_file_path, [self.label_file], remove=True)[0]

        self.logger.info("Reading the annotation file")
        annotations = {}
        annotation_count = self.get_line_count(label_file_path)
        with open(label_file_path, "r", encoding="utf-8") as f:
            for line in xtqdm(f, total=annotation_count):
                a = line.strip().split(" ")
                cat = a[0]
                document_id = a[1]
                if document_id in annotations:
                    annotations[document_id] += [cat]
                else:
                    annotations[document_id] = [cat]
        descs = ReutersNewsResource.read_descriptions(dir, self.kind)

        self.logger.info("Make annotated file")
        pathes = []
        for t in ["train", "test"]:
            file_path = os.path.join(dir, "{}_{}.txt".format(self.kind, t))
            self.logger.info("Annotating the {} file".format(t))
            data_path = os.path.join(dir, "lyrl2004_tokens_{}.csv".format(t))
            total_count = self.get_line_count(data_path)
            
            with open(file_path, "w", encoding="utf-8") as f:
                with open(data_path, "r", encoding="utf-8") as df:
                    for line in xtqdm(df, total=total_count):
                        doc_id, words = line.strip().split(",")
                        if doc_id in annotations:
                            ann = " ".join(annotations[doc_id])
                            if self.kind == "regions":
                                f.write("\t".join([ann, words]) + "\n")
                            else:
                                ps = [descs[a].parent for a in annotations[doc_id]]
                                ps = [p for p in ps if p not in ["Root", "None"]]
                                ps = list(set(ps))
                                ps = " ".join(ps)
                                f.write("\t".join([ann, ps, words]) + "\n")
            pathes.append(file_path)
            os.remove(data_path)
        os.remove(label_file_path)

        return pathes[0]

    def _download_file(self, url, file_path):
        r = requests.get(url)

        total_size = int(r.headers.get("content-length", 0))
        with open(file_path, "wb") as f:
            chunk_size = 1024
            limit = total_size / chunk_size
            for data in xtqdm(r.iter_content(chunk_size=chunk_size), total=limit, unit="B", unit_scale=True):
                f.write(data)

        return file_path

    def make_resource(self, data_root):
       return ReutersNewsResource(data_root, self.kind)

  
class ReutersNewsResource(Resource):

    def __init__(self, root, kind):
        columns = []
        target = ""
        if kind == "topics":
            columns = ["topic", "parent-topic", "words"]
            target = "topic"
        elif kind == "industries":
            columns = ["category", "2nd-level-topic", "words"]
            target = "category"
        elif kind == "regions":
            columns = ["region", "words"]
            target = "region"            

        super().__init__(
            root, 
            columns,
            target)
        
        self._resource = {
            "data": os.path.join(self.root, "{}_train.txt".format(kind)),
            "train": os.path.join(self.root, "{}_train.txt".format(kind)),
            "test": os.path.join(self.root, "{}_test.txt".format(kind)),
            "sample": os.path.join(self.root, "{}_train_samples.txt".format(kind))
        }
        self.descs = self.read_descriptions(self.root, kind)

    @classmethod
    def read_descriptions(cls, root, kind):
        Description = namedtuple("Description", ["code", "desc", "parent"])
        path = os.path.join(root, "rcv1.{}.txt".format(kind))
        descs = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                els = line.strip().split("\t")
                key = els[0]
                parent = "" if kind == "regions" else els[-2]
                desc = "" if kind == "regions" else els[-1]
                d = Description(key, desc, parent)
                descs[key] = d
        
        return descs
