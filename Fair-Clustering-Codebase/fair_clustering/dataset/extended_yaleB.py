import os
import sys
import numpy as np
from PIL import Image

from fair_clustering.dataset import ImageDataset


class ExtendedYaleB(ImageDataset):
    """ http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html """

    dataset_name = "Extended Yale B"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/extended_yaleB")
    file_url = {
        "CroppedYale.zip": "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip",
    }

    def __init__(self, download=True, center=True, resize=False):
        self.download_or_check_data(self.dataset_dir, self.file_url, download)

        dataset_dir = os.path.join(self.dataset_dir, "CroppedYale")
        dataset_folders = os.listdir(dataset_dir)
        all_img_paths = []
        for folder in dataset_folders:
            img_path = os.listdir(os.path.join(dataset_dir, folder))
            img_path = [os.path.join(dataset_dir, folder, p) for p in img_path if
                        p.endswith(".pgm") and p[-11:-4] != "Ambient"]
            all_img_paths.extend(img_path)

        if resize:
            X = [np.array(Image.open(p).resize((42, 48))) for p in all_img_paths]
        else:
            X = [np.array(Image.open(p)) for p in all_img_paths]

        X = np.asarray(X)
        X = np.reshape(X, (X.shape[0], -1))

        y = [int(p[-19:-17]) for p in all_img_paths]
        y = np.asarray(y)

        azimuth = [int(p[-11:-8]) for p in all_img_paths]
        elevation = [int(p[-6:-4]) for p in all_img_paths]
        s = [1 if (a >= 45 or e >= 45) else 0 for (a, e) in zip(azimuth, elevation)]
        s = np.asarray(s)

        super(ExtendedYaleB, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = ExtendedYaleB()
    X, y, s = dataset.data
    stat = dataset.stat
