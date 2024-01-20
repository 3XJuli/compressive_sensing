from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch


class MNISTDatasetFlattened(Dataset):
    def __init__(self, train, shuffle=False, n_measurements=None):
        if train:
            data = pd.read_csv("./data/mnist_train.csv")
        else:
            data = pd.read_csv("./data/mnist_test.csv")

        if shuffle:
            data = data.sample(frac=1)

        if n_measurements is not None:
            data = data.iloc[:n_measurements]

        images = data.drop(columns=["label"]).values
        self.labels = data["label"].values
        self.images = images.reshape((-1, 28 * 28)).astype("float32") / 255

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.Tensor([self.images[index]])

    def get_label(self, index):
        return self.labels[index]

    def plot_image_by_idx(self, idx):
        plt.imshow(self.images[idx], cmap="gray")
        plt.title(f"label: {self.labels[idx]}")

    def plot_image(image):
        plt.imshow(image, cmap="gray")
