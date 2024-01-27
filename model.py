import torch.nn as nn
import torch
from scipy.stats import ortho_group
from torch.nn.utils.parametrizations import orthogonal


class ISTA_RNN(nn.Module):
    def __init__(
        self, measurement_matrix, layers, lamb: float, tau: float, b_out: float, device
    ):
        """ """
        super(ISTA_RNN, self).__init__()
        self.layers = layers

        if layers <= 0:
            raise ValueError("layers must be greater than 0")
        self.measurement_matrix = measurement_matrix.to(device)

        self.mm_spectral_norm = torch.linalg.matrix_norm(self.measurement_matrix, ord=2)

        self.identity_matrix = torch.eye(self.measurement_matrix.shape[1]).to(device)

        self.lamb = lamb
        self.tau = tau
        self.shrinkage_factor = lamb * tau

        self.b_out = b_out

        if (tau * torch.pow(self.mm_spectral_norm, 2)) > 1:
            print(
                "Warning: Convergence is not guaranteed. (tau * mm_spectral_norm^2) = {0} > 1".format(
                    tau * torch.pow(self.mm_spectral_norm, 2)
                )
            )

        random_matrix = torch.randn(
            (self.measurement_matrix.shape[1], self.measurement_matrix.shape[1])
        )

        m = ortho_group.rvs(self.measurement_matrix.shape[1])

        random_matrix = torch.tensor(m, dtype=torch.float32)

        self.dictionary_layer = orthogonal(
            nn.Linear(
                self.measurement_matrix.shape[1],
                self.measurement_matrix.shape[1],
                bias=False,
                dtype=torch.float32,
            )
        )

        self.dictionary_layer.weight.data = random_matrix

    def shrink(self, x):
        return torch.sign(x) * torch.max(
            torch.abs(x) - self.shrinkage_factor, torch.zeros_like(x)
        )

    def shrink_relu(self, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - self.shrinkage_factor)

    def sigma(self, x):
        return self.b_out * x / torch.norm(x) if torch.norm(x) > self.b_out else x

    def loss(self, x, x_true):
        return ((x - x_true) ** 2).mean()

    def forward(self, y):
        f_i = self.shrink_relu(
            self.tau * (self.measurement_matrix @ self.dictionary_layer.weight).t() @ y
        )

        for i in range(1, self.layers):
            # perform ISTA step
            f_i = self.shrink_relu(
                (
                    self.identity_matrix
                    - self.tau
                    * self.dictionary_layer.weight.t()
                    @ self.measurement_matrix.t()
                    @ self.measurement_matrix
                    @ self.dictionary_layer.weight
                )
                @ f_i
                + self.tau
                * (self.measurement_matrix @ self.dictionary_layer.weight).t()
                @ y
            )

        return self.sigma(self.dictionary_layer.weight @ f_i)
