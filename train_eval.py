import torch
import time
import torch.nn.functional as F


def train(model, iterator, optimizer, loss, measurement_matrix, device):
    epoch_loss = 0

    model.train()
    optimizer.zero_grad()  # clear gradients first
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        start = time.time()
        # attention_mask, base_ids are already on device
        X = batch

        X = X.view(X.shape[0], X.shape[-1], -1).to(device)

        # encode
        Y = measurement_matrix @ X.double()

        # decode
        predictions = model(Y)

        loss_value = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
        )

        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss_value.item()
        end = time.time()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, loss, measurement_matrix, device):
    epoch_loss = 0

    model.eval()
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        # attention_mask, base_ids are already on device
        X = batch

        X = X.view(X.shape[0], X.shape[-1], -1).to(device)

        # encode
        Y = measurement_matrix @ X.double()

        predictions = model(Y)

        loss_value = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
        )

        epoch_loss = loss_value.item()

    return epoch_loss / len(iterator)
