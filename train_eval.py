import torch
import time
import torch.nn.functional as F


def train(model, iterator, optimizer, loss, device):
    epoch_loss = 0

    model.train()
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        start = time.time()
        # attention_mask, base_ids are already on device
        _, X = batch

        X = X.to(device)

        optimizer.zero_grad()  # clear gradients first

        predictions = model(X)

        loss_value = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
            model,
        )

        loss_value.backward()
        optimizer.step()

        epoch_loss += loss_value.item()
        end = time.time()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, loss, device):
    epoch_loss = 0

    model.eval()
    for i, batch in enumerate(
        iterator
    ):  # batch is simply a batch of ci-matricies as a tensor as x and y are the same
        # attention_mask, base_ids are already on device
        _, X = batch

        X = X.to(device)

        predictions = model(X)

        loss_value = loss(
            predictions.view(-1, 28 * 28),
            X.view(-1, 28 * 28),
            model,
        )

        epoch_loss = loss_value.item()

    return epoch_loss / len(iterator)
