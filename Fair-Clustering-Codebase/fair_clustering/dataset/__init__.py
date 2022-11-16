from fair_clustering.dataset.base import Dataset, TabDataset, ImageDataset
from fair_clustering.dataset.adult import Adult
from fair_clustering.dataset.compas import Compas
from fair_clustering.dataset.german import German
from fair_clustering.dataset.bank import Bank
from fair_clustering.dataset.credit import Credit
from fair_clustering.dataset.mnist_data import MNISTData, MNIST_USPS
from fair_clustering.dataset.extended_yaleB import ExtendedYaleB
from fair_clustering.dataset.office31 import Office31
from fair_clustering.dataset.mnist_usps import MNISTUSPS

__all__ = [
    "Dataset",
    "ExtendedYaleB",
    "Office31",
    "MNISTUSPS",
    "MNIST_USPS",
]
