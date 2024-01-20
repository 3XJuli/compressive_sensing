import torch.nn as nn
import torch
from scipy.stats import ortho_group


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

        random_matrix = torch.tensor(m, dtype=torch.float64)

        self.dictionary = nn.Parameter(random_matrix, requires_grad=True)

    def shrink(self, x):
        return torch.sign(x) * torch.max(
            torch.abs(x) - self.shrinkage_factor, torch.zeros_like(x)
        )

    def sigma(self, x):
        return self.b_out * x / torch.norm(x) if torch.norm(x) > self.b_out else x

    def loss(self, x, x_true):
        return ((x - x_true) ** 2).mean() + torch.norm(
            self.identity_matrix - self.dictionary.t() @ self.dictionary, p="fro"
        )

    def forward(self, y):
        f_i = self.shrink(
            self.tau * (self.measurement_matrix @ self.dictionary).t() @ y
        )

        for i in range(1, self.layers):
            f_i = self.shrink(
                (
                    self.identity_matrix
                    - self.tau
                    * self.dictionary.t()
                    @ self.measurement_matrix.t()
                    @ self.measurement_matrix
                    @ self.dictionary
                )
                @ f_i
                + self.tau * (self.measurement_matrix @ self.dictionary).t() @ y
            )

        return self.sigma(f_i)
