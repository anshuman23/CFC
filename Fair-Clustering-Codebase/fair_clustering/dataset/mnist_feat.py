import random
import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from fair_clustering.dataset import MNISTData


class Autoencoder(nn.Module):
    channel_mult = 4

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.channel_mult * 2, kernel_size=5),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 6, kernel_size=5),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult * 6, self.channel_mult * 4, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2, kernel_size=5),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ConvTranspose2d(self.channel_mult * 2, 1, kernel_size=5),
            nn.Tanh(),
        )

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        x = self.decoder(feat)
        if return_feat:
            return x, feat.flatten(start_dim=1)
        else:
            return x


if __name__ == "__main__":
    iter_ = 5000
    batch_size = 512
    device = torch.device("cuda:1")
    loss_func = nn.MSELoss()

    model = Autoencoder()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optim, step_size=iter_ // 4, gamma=0.1)
    X, y, s = MNISTData().data

    model.train()
    for i in range(iter_):
        optim.zero_grad()

        batch_rows_id = random.sample(range(0, X.shape[0] - 1), batch_size)
        batch_X = np.take(X, batch_rows_id, axis=0).reshape(-1, 28, 28)
        batch_X = batch_X[:, np.newaxis, :, :]
        batch_X = torch.from_numpy(batch_X).float()
        batch_X = batch_X.to(device)

        pred = model(batch_X)
        loss = loss_func(pred, batch_X)
        loss.backward()
        optim.step()
        scheduler.step()

        print("[%s/%s] loss: %.5f" % (i, iter_, loss.item()))

    model.eval()
    feat_list = []
    total_batch = (X.shape[0] // batch_size) + 1
    for i in range(total_batch):
        if i == total_batch - 1:
            batch_X = X[i * batch_size:, :]
        else:
            batch_X = X[i * batch_size:(i + 1) * batch_size]

        batch_X = torch.from_numpy(batch_X.reshape(-1, 28, 28)[:, np.newaxis, :, :]).float()
        batch_X = batch_X.to(device)

        with torch.no_grad():
            _, feat = model(batch_X, return_feat=True)
        feat_list.append(feat.cpu().numpy())

    feat = np.concatenate(feat_list, axis=0)
    with open("mnist_feat.npy", "wb") as f:
        np.save(f, feat)
