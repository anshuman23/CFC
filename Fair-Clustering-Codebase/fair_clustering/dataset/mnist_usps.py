import os
import sys
import scipy.io
import numpy as np

from fair_clustering.dataset import ImageDataset


class MNISTUSPS(ImageDataset):
    " MNIST and USPS dataset using SURF feature "

    dataset_name = "MNIST_USPS"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/mnist_usps")
    file_url = "https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA"

    def __init__(self, download=True, center=True):
        if not os.path.exists(os.path.join(self.dataset_dir, "MNIST_vs_USPS.mat")) and download:
            print(
                "Automatic download is not available for this dataset, please download the data files from %s and put them under %s" % (
                    self.file_url, self.dataset_dir))
            exit(1)

        mat = scipy.io.loadmat(os.path.join(self.dataset_dir, "MNIST_vs_USPS.mat"))
        s = np.concatenate([np.zeros(mat["X_src"].shape[1]), np.ones(mat["X_tar"].shape[1])])
        X = np.concatenate([mat["X_src"].T, mat["X_tar"].T], axis=0)
        y = np.concatenate([mat["Y_src"].squeeze(), mat["Y_tar"].squeeze()], axis=0)

        super(MNISTUSPS, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = MNISTUSPS()
    X, y, s = dataset.data
    stat = dataset.stat

    print(stat)
