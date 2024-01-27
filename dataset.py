from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch


class MNISTDatasetFlattened(Dataset):
    def __init__(
        self,
        train,
        measurement_matrix=None,
        shuffle=False,
        training_size=None,
        device=torch.device("cpu"),
    ):
        if train:
            data = pd.read_csv("./data/mnist_train.csv")
        else:
            data = pd.read_csv("./data/mnist_test.csv")

        if shuffle:
            data = data.sample(frac=1)

        if training_size is not None:
            data = data.iloc[:training_size]

        if measurement_matrix is not None:
            self.measurement_matrix = measurement_matrix

        images = data.drop(columns=["label"]).values
        self.labels = data["label"].values
        self.images = images.reshape((-1, 28 * 28)).astype("float32") / 255

        self.images = torch.tensor(
            self.images, dtype=torch.float32, device=device
        ).view(self.images.shape[0], -1, 1)

        self.measurements = self.measurement_matrix @ self.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.measurements[index]

    def get_label(self, index):
        return self.labels[index]

    def plot_image_by_idx(self, idx):
        plt.imshow(self.images[idx], cmap="gray")
        plt.title(f"label: {self.labels[idx]}")

    def plot_image(self, image):
        plt.imshow(image, cmap="gray")
