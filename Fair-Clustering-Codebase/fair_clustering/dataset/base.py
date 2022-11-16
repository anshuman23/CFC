import os
import gdown
import tarfile
import warnings
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Sequence
from collections import Counter
import urllib.request
import numpy as np
from zipfile import ZipFile
from sklearn import preprocessing


class Dataset(ABC):
    """ Abstract base class for fair clustering datasets """

    def download_or_check_data(self, dir: str, file_url: Dict[str, str], download: bool,
                               use_gdown: bool = False) -> None:
        """ Download data or check data existence """

        if not os.path.exists(dir):
            os.makedirs(dir)

        def check_file_exist(dir: str, file_url: Dict[str, str]) -> Dict[str, bool]:
            flag = {f: False for f in file_url.keys()}
            for file_name in file_url.keys():
                if os.path.isfile(os.path.join(dir, file_name)):
                    flag[file_name] = True
            return flag

        def raise_not_exist(exist_flag: Dict[str, bool]) -> None:
            for file_name, exist in exist_flag.items():
                if not exist:
                    print("%s does not exist, consider set `download` to Ture" % file_name)

            if False in exist_flag.values():
                exit(1)
            else:
                return

        def download_data(dir: str, file_url: Dict[str, str], exist_flag: [str, bool], use_gdown: bool) -> None:
            for file_name, url in file_url.items():
                if not exist_flag[file_name]:
                    print("Download %s from %s" % (file_name, url))
                    if use_gdown:
                        gdown.download(url, os.path.join(dir, file_name), quiet=False)
                    else:
                        urllib.request.urlretrieve(url, os.path.join(dir, file_name))

                if file_name.endswith(".zip"):
                    zf = ZipFile(os.path.join(dir, file_name), 'r')
                    zf.extractall(dir)
                    zf.close()

                if file_name.endswith(".tar.gz"):
                    f = tarfile.open(os.path.join(dir, file_name))
                    f.extractall(dir)
                    f.close()

            return

        exist_flag = check_file_exist(dir, file_url)
        if not download:
            raise_not_exist(exist_flag)
        else:
            download_data(dir, file_url, exist_flag, use_gdown)

        return

    def validate(self) -> None:
        """ Validation """

        self.X = self.X.astype(np.float64)
        self.y = self.y.astype(np.int)
        self.s = self.s.astype(np.int)

        assert len(np.unique(self.y)) > 1, "Only one class exists in the dataset"
        assert np.any(self.y) >= 0, "Class label should be an integer larger or equal to zero"
        assert np.any(self.s) in (1, 0), "Sensitive attribute should be a binary value"

        return

    def save(self, dir: str):
        np.save(os.path.join(dir, "X.npy"), self.X)
        np.save(os.path.join(dir, "y.npy"), self.y)
        np.save(os.path.join(dir, "s.npy"), self.s)

        return

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.X, self.y, self.s)

    @property
    def stat(self) -> Dict:
        y, s = self.y, self.s
        y_stat, s_stat = Counter(y), Counter(s)

        stat = {}
        for y_label, num in y_stat.items():
            stat["#class='%s'" % y_label] = num
        for s_label, num in s_stat.items():
            stat["#sensitive='%s'" % s_label] = num

        s_ratio = float(stat["#sensitive='%s'" % tuple(s_stat.keys())[0]]) / float(
            stat["#sensitive='%s'" % tuple(s_stat.keys())[1]])
        s_ratio = 1. / s_ratio if s_ratio > 1 else s_ratio
        stat.update({"sensitive_ratio": s_ratio})

        return stat

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    def __repr__(self):
        return "Fair clustering dataset: %s" % self.dataset_name


class TabDataset(Dataset):
    """

    Base class for fair clustering on tabular datasets
    Currently only support binary sensitive attributes

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    df : pd.DataFrame
        Dataset as a DataFrame.

    sensitive : str
        Name of the sensitive attribute.

    exclude_s : bool
        Whether exclude the sensitive attribute from data for clustering.

    center : bool
        Whether normalize the data using sklearn preprocessing.StandardScaler

    numerical_s : bool
        Set to True if the sensitive attribute is a numerical feature.

    s_thr : float, default=None
        The threshold to binarize a numerical sensitive attribute.

    Attributes
    ----------
    info : Dict
        The meta data for the dataset.

    data: Tuple[np.ndarray, np.ndarray, np.ndarray]
        The data tuple including X, y, and s.
        X : np.ndarray of shape (n_samples, n_feat), the data for clustering
        y : np.ndarray of shape (n_samples,), the target label of X
        s : np.ndarray of shape (n_samples,), the sensitive attribute of X

    stat: Dict
        Statistics for data and its sensitive attributes.

    """

    def __init__(
            self,
            df: pd.DataFrame,
            sensitive: str,
            exclude_s: bool,
            center: bool,
            numerical_s: bool,
            s_thr: float = None,
    ):
        self.sensitive = sensitive
        self.exclude_s = exclude_s
        self.center = center
        self.numerical_s = numerical_s
        self.s_thr = s_thr

        df.dropna(inplace=True)
        self.y, self.ori_y = self.encode_single_categorical_feat(df, self.label_name)
        if not self.numerical_s:
            self.s, self.ori_s = self.encode_single_categorical_feat(df, self.sensitive)
        else:
            self.s, self.ori_s = self.encode_single_numerical_feat(df, self.sensitive)

        df.drop(columns=self.drop_feat, inplace=True)
        df.drop(labels=self.label_name, axis=1, inplace=True)
        if self.exclude_s:
            df.drop(labels=self.sensitive, axis=1, inplace=True)

        self.X = self.one_hot_encoding(df)
        self.validate()

        if self.center:
            self.X = self.scale(self.X)

        if self.numerical_s:
            self.default_mappings.update({sensitive: {">=%.f" % self.s_thr: 1, "<%.f" % self.s_thr: 0}})

        self._metadata = {
            "dataset_name": self.dataset_name,
            "label_name": self.label_name,
            "#instance": self.X.shape[0],
            "#dim": self.X.shape[1],
            "#class": len(np.unique(self.y)),
            "sensitive_attribute": self.sensitive,
            "exclude_s": self.exclude_s,
            "center": self.center,
            "label_mapping": self.default_mappings[self.label_name],
            "sensitive_mapping": self.default_mappings[self.sensitive],
        }

    def encode_single_categorical_feat(self, df: pd.DataFrame, feat_name: str) -> Tuple[np.ndarray, np.array]:
        """ Extract and encode a single categorical feature using default mappings """

        assert feat_name in self.default_mappings

        feat = df[feat_name].to_numpy().copy()
        ori_feat = feat.copy()
        for i in range(len(feat)):
            feat[i] = self.default_mappings[feat_name][feat[i]]

        return feat, ori_feat

    def encode_single_numerical_feat(self, df: pd.DataFrame, feat_name: str) -> Tuple[np.ndarray, np.array]:
        """ Extract and encode a single numerical feature using default mappings """

        feat = df[feat_name].to_numpy().astype(np.float64).copy()
        ori_feat = feat.copy()
        if self.s_thr == None:
            warnings.warn("Threshold is not set for numerical sensitive attribute, using the median value.")
            self.s_thr = np.median(feat)
        else:
            assert self.s_thr > np.min(feat), \
                "Sensitive threshold should be larger than the minimum value %.f." % np.min(feat)
            assert self.s_thr <= np.max(feat), \
                "Sensitive threshold should be equal or smaller than the maximum value %.f." % np.max(feat)

        for i in range(len(feat)):
            feat[i] = feat[i] >= self.s_thr

        return feat, ori_feat

    def one_hot_encoding(self, df: pd.DataFrame) -> np.ndarray:
        for c in df.columns:
            if c in self.categorical_feat:
                column = df[c]
                df.drop(labels=c, axis=1, inplace=True)

                if c in self.default_mappings:
                    mapping = self.default_mappings[c]
                else:
                    unique_values = pd.unique(column)
                    mapping = {v: i for i, v in enumerate(unique_values)}

                n = len(set(mapping.values()))
                if n > 2:
                    for i in range(n):
                        col_name = '{}.{}'.format(c, i)
                        col_value = [1. if mapping[e] == i else 0. for e in column.to_numpy()]
                        df[col_name] = col_value
                else:
                    col_value = [mapping[e] for e in column.to_numpy()]
                    df[c] = col_value

        return df.to_numpy()

    def scale(self, X: np.ndarray) -> np.ndarray:
        """ Normalization per column """
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        return X

    @property
    @abstractmethod
    def categorical_feat(self) -> Sequence[str]:
        return self.categorical_feat

    @property
    @abstractmethod
    def drop_feat(self) -> Sequence[str]:
        return self.drop_feat

    @property
    @abstractmethod
    def avail_s(self) -> Sequence[str]:
        return self.avail_s

    @property
    @abstractmethod
    def label_name(self) -> str:
        return self.label_name

    @property
    @abstractmethod
    def default_mappings(self) -> Dict[str, Dict[str, int]]:
        return self.default_mappings


class ImageDataset(Dataset):
    """

    Base class for fair clustering on image datasets
    Currently only support binary sensitive attributes

    """

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            s: np.ndarray,
            center: bool,
    ):
        self.center = center

        self.X = X
        self.y = y
        self.s = s

        self.validate()

        if self.center:
            self.X = self.scale(self.X)

        self._metadata = {
            "dataset_name": self.dataset_name,
            "#instance": self.X.shape[0],
            "#dim": self.X.shape[1],
            "#class": len(np.unique(self.y)),
            "center": self.center,
        }

    def scale(self, X: np.ndarray, eps=1e-5) -> np.ndarray:
        """ Normalize to [-1, 1] """
        mean = np.mean(X)
        std = np.std(X)
        X = (X - mean) / (std + eps)

        return X
