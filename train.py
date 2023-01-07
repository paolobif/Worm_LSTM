import os
import torch
import logging
from tqdm import tqdm
from datetime import date
import matplotlib.pyplot as plt

from utils.data_utils import LstmLoader
from utils.training_utils import test_model, convert_confusion
from model import ResnetLSTM

import torch.nn as nn
import torch.optim as optim


def check_save(path):
    if os.path.exists(path):
        res = input("Save Path Exists - Overwrite (y or n): ")
        if res != "y":
            print("Exiting Training")
            raise SystemExit
    else:
        os.makedirs(path)


def build_dataset(data_loader, split=0.15):
    """Takes dataloader and partitions it based on specified split
    params.

    Args:
        data_loader: Worm LSTM dataloader object.

    Returns:
        test, train: respective data loader objects.
    """
    test_len = int(len(data_loader) * split)
    train_len = len(data_loader) - test_len

    print(f"Total Data: {len(data_loader)}")
    print(f"Test split: {test_len}")
    print(f"Train split: {train_len}")
    log_dataset(len(data_loader), train_len, test_len)

    test_data, train_data = torch.utils.data.random_split(data_loader, (test_len, train_len))
    return test_data, train_data


def log_dataset(total: int, train: int, test: int):
    logging.info(f"Dataset info:\n\ttotal: {total}, train: {train}, test: {test}")


def log_training(params):
    log_str = f"Epoch: {params['epoch']} \
              - Acc: {params['accuracy']} \
              - A_acc: {params['alive']} \
              - D_acc: {params['dead']} \
              - Loss: {params['loss']}"
    print(log_str)
    logging.info(log_str)


def plot_training(loss_totals, test_totals):
    f, ax = plt.subplots()
    ax.plot(test_totals, color="red", label="Test Accuracy")
    ax.set_ylabel("Test Accuracy")
    ax2 = ax.twinx()
    ax2.plot(loss_totals, color="blue", label="Loss")
    ax2.set_ylabel("Loss")
    ax.legend()
    ax2.legend()
    return f


def train(train_data, test_data, epochs: int, device, save_path):
    loss_totals = []
    test_totals = []

    for epoch in range(epochs):
        loss_total = 0
        for series, label in tqdm(train_data):
            model.zero_grad()

            series = series.to(device)
            label = label.to(device)
            #  label = label.view(1,2)

            output = model(series).squeeze()
            #  print(output, label)
            loss = criterion(output, label.float())
            loss_total += loss
            #  print(output, label, loss)
            loss.backward()
            optimizer.step()

        # Confusion arithmetic.
        matrix = test_model(model, test_data, device)
        accuracy, d_accuracy, a_accuracy = convert_confusion(matrix)
        loss = loss_total / len(train_data)

        log_params = {
            "epoch": epoch,
            "accuracy": round(accuracy, 4),
            "alive": round(a_accuracy, 4),
            "dead": round(d_accuracy, 4),
            "loss": loss
        }
        log_training(log_params)

        loss_totals.append(loss_total / len(train_data))
        test_totals.append(accuracy)
        figure = plot_training(loss_totals, test_totals)
        figure.savefig(os.path.join(save_path, "plot.png"))

        # Save weights
        torch.save(model.state_dict(), os.path.join(save_path, f"weights{epoch}.pt"))
        print("weights saved")


if __name__ == "__main__":
    data_path = "data/training_real"
    weights_path = "weights/training_f2"

    # File checks and logging.
    check_save(weights_path)
    logging.basicConfig(
        filename=os.path.join(weights_path, "train.log"),
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    data_loader = LstmLoader(data_path)

    # Settings
    training_split = 0.15  # What fraction to use as test data.

    learning_rate = 0.0001  # Learning rate.

    epochs = 100  # How many epochs to run.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(
        f"""\n\nStarting training at {date.today()}
            Source: {data_path}
            Save: {weights_path}
            Epochs: {epochs}
            Device: {device}
        """
    )
    # Setup training Data
    test_data, train_data = build_dataset(data_loader)

    # Setup Model
    model = ResnetLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    train(train_data, test_data, epochs, device, weights_path)
