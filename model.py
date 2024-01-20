import torch.nn as nn
import torch


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

        self.mm_spectral_norm = torch.nn.utils.spectral_norm(self.measurement_matrix)

        self.identity_matrix = torch.eye(self.measurement_matrix.shape[0]).to(device)

        self.lamb = lamb
        self.tau = tau
        self.shrinkage_factor = lamb * tau

        self.b_out = b_out

        if (tau * torch.pow(self.mm_spectral_norm, 2)) > 1:
            print("Warning: Convergence is not guaranteed.")

        random_matrix = torch.randn(self.measurement_matrix.shape)

        svd = torch.svd(random_matrix)

        random_orthogonal_matrix = svd.U @ torch.diag(svd.S)

        self.dictionary = nn.Parameter(random_orthogonal_matrix).to(device)

    def shrink(self, x):
        return torch.sign(x) * torch.max(torch.abs(x) - self.lamb, torch.zeros_like(x))

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
