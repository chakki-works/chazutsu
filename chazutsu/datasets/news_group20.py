import os
import re
import tarfile
import shutil
from chazutsu.datasets.framework.xtqdm import xtqdm
from chazutsu.datasets.framework.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class NewsGroup20(Dataset):

    def __init__(self, group_filter=()):
        super().__init__(
            name="20 Newsgroups",
            site_url="http://qwone.com/~jason/20Newsgroups/",
            download_url="http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz",
            description="news article and its comments that are categorized by 20 group."
            )
        
        self.group_filter = group_filter
        self._mail_pattern = re.compile("[\w|\.]+@[\w|\.]+")

    def extract(self, path):
        dir, file_name = os.path.split(path)
        work_dir = os.path.join(dir, "tmp")
        newsgroup20_path = os.path.join(dir, "newsgroup20.txt")

        if not os.path.isdir(work_dir):
            with tarfile.open(path) as t:
                t.extractall(path=work_dir)

        dataset_path = os.path.join(work_dir, "20news-18828")
        with open(newsgroup20_path, mode="wb") as f:
            for gp in os.listdir(dataset_path):
                group_path = os.path.join(dataset_path, gp)
                if not os.path.isdir(group_path):
                    continue
                if len(self.group_filter) > 0 and gp not in self.group_filter:
                    continue

                self.logger.info("Extracting {} news data.".format(gp))
                for news in xtqdm(os.listdir(group_path)):
                    group_name = gp
                    category_name = self.get_category(gp)
                    subject, author, text = self.parse(path=os.path.join(group_path, news))
                    ln = "\t".join([
                        group_name,
                        category_name,
                        subject,
                        author,
                        text
                    ]) + "\n"
                    f.write(ln.encode("utf-8"))

        # remove files
        os.remove(path)
        shutil.rmtree(work_dir)

        return newsgroup20_path

    def make_resource(self, data_root):
        return Resource(data_root, columns=["group", "group-category", "subject", "author", "text"], target="group")

    def get_category(self, group_name):
        g = group_name
        if g.startswith("talk."):
            g = g.replace("talk.", "")

        cats = group_name.split(".")
        category_name = cats[0]
        if g in ["alt.atheism", "soc.religion.christian"]:
            category_name = "religion"
        
        return category_name
    
    def parse(self, path="", raw_text=""):
        body = raw_text

        if body:
            body = body.split("\n")
        elif path:
            with open(path, errors="ignore", encoding="utf-8") as f:
                body = f.readlines()
        else:
            raise Exception("Can not get parse target text.")
        
        def strip(s):
            _s = s
            for c in ["<", ">", "^", "-", "(", ")", "*"]:
                _s = _s.replace(c , "")

            _s = re.sub(self._mail_pattern, "", _s)
            _s = _s.replace("\t", " ")
            _s = _s.strip()
            if not _s:
                return ""
            elif _s.endswith("writes:"):
                return ""
            elif _s.startswith("Archive-name:"):
                return ""
            elif _s.startswith("Alt-atheism-archive-name:"):
                return ""
            elif _s.startswith("Last-modified:"):
                return ""
            elif _s.startswith("Version:"):
                return ""
            else:
                return _s

        body = [s for s in [strip(s) for s in body] if s]
        subject = ""
        author = ""
        text = ""

        for i, s in enumerate(body):
            els = s.split(":")
            if len(els) < 2:
                continue

            if els[0].startswith("From"):
                author = ":".join(els[1:]).strip()
            elif els[0].startswith("Subject"):
                subject = ":".join(els[1:]).strip()
            
            if author and subject:
                break

            if i > 2:
                break  # can not find out
        
        text = " ".join(body[2:])

        return subject, author, text
