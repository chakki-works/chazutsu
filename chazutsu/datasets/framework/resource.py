import os
import pandas as pd


class Resource():

    def __init__(self, 
        root,
        columns=None,
        target="",
        train_file_suffix="_train",
        test_file_suffix="_test",
        sample_file_suffix="_samples"):

        self.root = root
        self.columns = columns
        self.target = target
        self.path = ""
        self.train_file_path = ""
        self.test_file_path = ""
        self.sample_file_path = ""

        if target and target not in columns:
            raise Exception("Target have to be selected from columns.")

        self._file_pathes = []
        for f in os.listdir(self.root):
            p = os.path.join(self.root, f)
            if not os.path.isfile(p):
                continue
            
            n, e = os.path.splitext(f)
            if n.endswith(train_file_suffix):
                self.train_file_path = p
            elif n.endswith(test_file_suffix):
                self.test_file_path = p
            elif n.endswith(sample_file_suffix):
                self.sample_file_path = p
            elif not self.path:
                self.path = p
        
        if not self.path:
            self.path = self.train_file_path

    def data(self, split_target=False):
        if self.path:
            return self._to_pandas(self.path, split_target)
        elif self.train_file_path:
            return self._to_pandas(self.train_file_path, split_target)

    def train_data(self, split_target=False):
        return self._to_pandas(self.train_file_path, split_target)

    def test_data(self, split_target=False):
        return self._to_pandas(self.test_file_path, split_target)

    def sample_data(self, split_target=False):
        return self._to_pandas(self.sample_file_path, split_target)

    def _to_pandas(self, path, split_target):
        df = pd.read_table(path, header=None, names=self.columns)

        if not split_target:
            return df
        elif self.target and self.target in df.columns:
            target = df[self.target]
            df.drop(self.target, axis=1, inplace=True)
            return target, df
        else:
            return df
