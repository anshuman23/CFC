import os
import sys
import numpy as np
import scipy.io

from fair_clustering.dataset import ImageDataset


class Office31(ImageDataset):
    """ https://faculty.cc.gatech.edu/~judy/domainadapt/ """

    dataset_name = "Office 31"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/office31")
    file_url = {
        "domain_adaptation_features_20110616.tar.gz": "https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WTVSd2FIcW4wRTA&export=download",
        "office31_resnet50.zip": "https://wjdcloud.blob.core.windows.net/dataset/office31_resnet50.zip",
    }
    all_domain = ["amazon", "dslr", "webcam"]

    label = {
        'speaker': 0,
        'mug': 1,
        'trash_can': 2,
        'pen': 3,
        'bike_helmet': 4,
        'mouse': 5,
        'paper_notebook': 6,
        'desk_lamp': 7,
        'keyboard': 8,
        'printer': 9,
        'mobile_phone': 10,
        'tape_dispenser': 11,
        'ring_binder': 12,
        'desktop_computer': 13,
        'phone': 14,
        'desk_chair': 15,
        'punchers': 16,
        'back_pack': 17,
        'letter_tray': 18,
        'headphones': 19,
        'projector': 20,
        'bottle': 21,
        'scissors': 22,
        'stapler': 23,
        'bike': 24,
        'laptop_computer': 25,
        'monitor': 26,
        'file_cabinet': 27,
        'bookcase': 28,
        'calculator': 29,
        'ruler': 30,
    }

    def __init__(self, exclude_domain: str, download=True, center=True, use_feature=False):
        assert exclude_domain in self.all_domain, "Exclude domain for %s dataset should be %s" % (
            self.dataset_name, " or ".join([s for s in self.all_domain]))
        self.download_or_check_data(self.dataset_dir, self.file_url, download, use_gdown=True)

        domains = self.all_domain
        domains.remove(exclude_domain)

        if not use_feature:
            X, y, s = [], [], []
            for i, domain in enumerate(domains):
                domain_dir = os.path.join(self.dataset_dir, "domain_adaptation_features", domain, "interest_points")
                category = os.listdir(domain_dir)
                for c in category:
                    img_paths = os.listdir(os.path.join(domain_dir, c))
                    img_paths = [os.path.join(domain_dir, c, p) for p in img_paths if p[-17:-14] == "800"]
                    imgs = [np.asarray(scipy.io.loadmat(p)["histogram"]).reshape(-1) for p in img_paths]

                    X.extend(imgs)
                    y.extend([self.label[c] for _ in range(len(imgs))])
                    s.extend([i for _ in range(len(imgs))])

            X, y, s = np.asarray(X), np.asarray(y), np.asarray(s)
        else:
            domain_1 = np.genfromtxt(
                os.path.join(self.dataset_dir, "resnet50_feature", "%s_%s.csv" % (domains[0], domains[0])),
                delimiter=",")
            domain_2 = np.genfromtxt(
                os.path.join(self.dataset_dir, "resnet50_feature", "%s_%s.csv" % (domains[1], domains[1])),
                delimiter=",")

            X = np.concatenate([domain_1[:, :-1], domain_2[:, :-1]], axis=0)
            y = np.concatenate([domain_1[:, -1], domain_2[:, -1]], axis=0)
            s = np.concatenate([np.zeros(domain_1.shape[0]), np.ones(domain_2.shape[0])], axis=0)

        super(Office31, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = Office31(exclude_domain="amazon", use_feature=True)
    X, y, s = dataset.data
    stat = dataset.stat
