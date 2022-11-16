import os
import sys
import numpy as np
import scipy.io
from mnist import MNIST

from fair_clustering.dataset import ImageDataset


class MNISTData(ImageDataset):
    """ http://yann.lecun.com/exdb/mnist/ """

    dataset_name = "MNIST"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/mnist")
    file_url = {
        "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    }

    def __init__(self, download=True, center=True, full=True):
        self.download_or_check_data(self.dataset_dir, self.file_url, download)

        mndata = MNIST(MNISTData.dataset_dir)
        mndata.gz = True
        images, labels = mndata.load_training()
        images, labels = np.asarray(images), np.asarray(labels)

        if full is not True:
            images = images[:1000]
            labels = labels[:1000]

        images_rev = 255 - images

        X = np.concatenate([images, images_rev], axis=0)
        y = np.concatenate([labels, labels], axis=0)
        s = np.asarray([0 for _ in range(images.shape[0])] + [1 for _ in range(images_rev.shape[0])])

        super(MNISTData, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


class MNIST_USPS(ImageDataset):
    """
    http://yann.lecun.com/exdb/mnist/
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps
    """

    dataset_name = "MNIST_USPS"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/mnist_usps")

    # download link: https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA

    def __init__(self, center=True):
        # change to USPS_vs_MNIST.mat if you want
        data = scipy.io.loadmat(os.path.join(self.dataset_dir, "MNIST_vs_USPS.mat"))

        X_1 = data["X_src"].T
        X_2 = data["X_tar"].T
        y_1 = data["Y_src"].squeeze()
        y_2 = data["Y_tar"].squeeze()

        X = np.concatenate([X_1, X_2], axis=0)
        y = np.concatenate([y_1, y_2], axis=0)
        s = np.asarray([0 for _ in range(X_1.shape[0])] + [1 for _ in range(X_2.shape[0])])

        super(MNIST_USPS, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    # dataset = MNISTData(full=False)
    # X, y, s = dataset.data
    # stat = dataset.stat
    #
    # print(stat)

    dataset = MNIST_USPS()
