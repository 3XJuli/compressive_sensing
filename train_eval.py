import torch
import time
import torch.nn.functional as F


def train(model, iterator, optimizer, loss, measurement_matrix, device):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0

    model.train()
    optimizer.zero_grad()  # clear gradients first
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        # attention_mask, base_ids are already on device
        X, Y = batch
        # decode
        predictions = model(Y)

        mse_loss = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
        )

        loss_value = mse_loss  # + 0.001 * orthogonality_regularization

        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print(model.dictionary.norm(p="fro"))

        epoch_loss += loss_value.item()

        epoch_mse_loss += mse_loss.item()

        epoch_mae_loss += F.l1_loss(
            predictions.view(-1, 28 * 28), X.view(-1, 28 * 28)
        ).item()

    return (
        epoch_loss / len(iterator),
        epoch_mae_loss / len(iterator),
        epoch_mse_loss / len(iterator),
    )


def evaluate(model, iterator, loss, measurement_matrix, device):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0

    model.eval()
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        # attention_mask, base_ids are already on device
        X, Y = batch
        # decode
        predictions = model(Y)

        mse_loss = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
        )

        loss_value = mse_loss

        epoch_loss += loss_value.item()

        epoch_mse_loss += mse_loss.item()

        epoch_mae_loss += F.l1_loss(
            predictions.view(-1, 28 * 28), X.view(-1, 28 * 28)
        ).item()

    return (
        epoch_loss / len(iterator),
        epoch_mae_loss / len(iterator),
        epoch_mse_loss / len(iterator),
    )
