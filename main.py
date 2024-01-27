from re import M
from dataset import MNISTDatasetFlattened
from train_eval import train, evaluate
import torch.nn.functional as F

import sys

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time

from model import ISTA_RNN

import os
import argparse
import wandb
import random
import numpy as np
import sys

from functools import partial
from enum import Enum

# need this seed for the lookup (as data is randomly shuffled)
random.seed(1234)
np.random.seed(1234)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def generate_during_training(model, samples, epoch=None):
    for X, Y in samples:
        X_hat = model(Y)
        true_img = wandb.Image(X.view(28, 28), caption="original sample")
        pred_img = wandb.Image(X_hat.view(28, 28), caption="reconstructed sample")
        wandb.log({"original": true_img, "reconstructed": pred_img, "epoch": epoch})


def generate_measurement_matrix(N, n):
    return torch.randn(n, N, dtype=torch.float32) * torch.sqrt(torch.tensor(1.0 / n))


def main():
    output_dir = "./output/"

    # this is whats actually used
    BS = 256
    SEED = 1234
    N_EPOCHS = 40
    lamb = 0.5
    learning_rate = 0.002
    layers = 100
    # image dimension
    N = 28 * 28

    generate_every_n_epochs = 5
    generate_n_meshes = 3

    training_size = BS * 10

    # number of measurements
    n_measurements = 200

    device = "auto"
    log_wandb = 1

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    run_params_name = f"bs_{BS}_lr_{learning_rate}_n_{n_measurements}_epochs_{N_EPOCHS}_lamb_{lamb}_layers_{layers}_training_size_{training_size}"

    output_file = run_params_name

    use_checkpoint = True

    checkpoint_path = f"{output_dir}/cp_{output_file}.pt"
    random.seed(SEED)
    np.random.seed(SEED)

    if log_wandb:
        project = "seminar-compressive-sensing"

        wandb.init(
            project=project,
            entity="julian-streit1",
            name=run_params_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": BS,
                "SEED": SEED,
                "epochs": N_EPOCHS,
                "use_checkpoint": use_checkpoint,
                "checkpoint_path": checkpoint_path,
                "generate_every_n_epochs": generate_every_n_epochs,
                "generate_n_meshes": generate_n_meshes,
                "n_measurements": n_measurements,
                "lamb": lamb,
            },
        )

    measurement_matrix = generate_measurement_matrix(N, n_measurements).to(device)

    train_dataset = MNISTDatasetFlattened(
        train=True,
        measurement_matrix=measurement_matrix,
        shuffle=True,
        device=device,
        training_size=training_size,
    )
    val_dataset = MNISTDatasetFlattened(
        train=False, measurement_matrix=measurement_matrix, shuffle=False, device=device
    )

    train_dl = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=BS, shuffle=False)

    samples = []
    for i in range(generate_n_meshes):
        samples.append(train_dataset.__getitem__(i))

    tau = 1 / (torch.linalg.matrix_norm(measurement_matrix, ord=2) ** 2)

    print(f"N: {N}, n: {n_measurements}")

    model = ISTA_RNN(
        measurement_matrix, layers=layers, lamb=lamb, tau=tau, b_out=1, device=device
    )

    best_val_mse_loss = float("inf")

    loss = model.loss

    model = model.to(device)
    print(f"model created..")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"start training..")

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_mae_loss, train_mse_loss = train(
            model,
            train_dl,
            optimizer,
            loss,
            measurement_matrix,
            device,
        )
        val_loss, val_mae_loss, val_mse_loss = evaluate(
            model, val_dl, loss, measurement_matrix, device
        )

        if log_wandb and epoch % generate_every_n_epochs == 0:
            generate_during_training(model, samples, epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_mse_loss < best_val_mse_loss:
            best_val_mse_loss = val_mse_loss
            torch.save(model.state_dict(), f"{output_dir}/{output_file}.pt")

        if log_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mae_loss": train_mae_loss,
                    "train_mse_loss": train_mse_loss,
                    "normalized_train_mse_loss": train_mse_loss / N,
                    "normalized_val_mse_loss": val_mse_loss / N,
                    "val_mae_loss": val_mae_loss,
                    "val_mse_loss": val_mse_loss,
                    "best_val_mse_loss": best_val_mse_loss,
                    "epoch": epoch,
                }
            )

        epoch_msg = (
            f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s \n"
            + f"Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Best Val Loss: {best_val_mse_loss:.3f} \n"
            + f"Train MAE Loss: {train_mae_loss:.3f} | Val. MAE Loss: {val_mae_loss:.3f} \n"
            + f"Train MSE Loss: {train_mse_loss:.3f} | Val. MSE Loss: {val_mse_loss:.3f}"
        )
        print(epoch_msg)
        f = open(f"{output_dir}/{output_file}.txt", "a")
        f.write(epoch_msg)
        f.close()

        # save model and optimizer every 20 epochs
        if epoch % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mae_loss": train_mae_loss,
                    "train_mse_loss": train_mse_loss,
                    "val_mae_loss": val_mae_loss,
                    "val_mse_loss": val_mse_loss,
                    "best_val_loss": best_val_mse_loss,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    # load param
    main()
