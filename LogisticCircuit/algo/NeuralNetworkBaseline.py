import math

import logging

import copy
from time import perf_counter

import numpy as np
import torch
from torch import nn, optim
from typing import List, Optional, Tuple

from torch.utils.data import TensorDataset, DataLoader

from ..util.DataSet import DataSet


class NeuralNetworkRegressor(nn.Module):
    def __init__(self, input_size: int, loss_fn):
        super().__init__()
        middleSize = math.floor(input_size / 2)
        endSize = math.floor(input_size / 4)
        self.layers = nn.Sequential(
            nn.Linear(input_size, middleSize),
            nn.ReLU(),
            nn.Linear(middleSize, endSize),
            nn.ReLU(),
            nn.Linear(endSize, 1),
        )
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor):
        return self.layers(x).squeeze()

    def calculate_error(self, dataset: DataSet) -> float:
        return float(self.loss_fn(self(torch.from_numpy(dataset.images)), dataset.labels))


def learn_neural_network(
        train: DataSet,
        valid: Optional[DataSet] = None,
        max_iter: int = 200,
        validate_every: int = 10,
        patience: int = 2,
        seed: int = 1337
) -> Tuple[NeuralNetworkRegressor, List[float]]:

    # TODO: anymore work for seeds?
    torch.manual_seed(seed)

    error_history: List[float] = []

    network = NeuralNetworkRegressor(train.images.shape[1], nn.MSELoss())
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    logging.info(f"Network has {network.parameters()} parameters")

    network.eval()
    train_acc = network.calculate_error(train)
    logging.info(f" error: {train_acc:.5f}")
    error_history.append(train_acc)

    torchDataset = TensorDataset(torch.from_numpy(train.images), train.labels.float())
    dataLoader = DataLoader(torchDataset, batch_size=10, shuffle=True)

    logging.info("Start network learning.")
    sl_start_t: float = perf_counter()

    valid_best = +np.inf
    best_parameters = copy.deepcopy(network.state_dict())
    c_patience = 0
    for i in range(max_iter):
        cur_time = perf_counter()

        network.train()

        # forward pass
        totalLoss = 0
        for batch_idx, (images, labels) in enumerate(dataLoader):
            optimizer.zero_grad()
            # print(images.dtype)
            # print(labels.dtype)
            y_pred = network(images)
            loss = network.loss_fn(y_pred, labels)
            # backwards pass
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        logging.info(f"done iter {i+1}/{max_iter} in {perf_counter() - cur_time} secs")
        logging.info(f"\terror: {totalLoss/len(dataLoader):.5f}")

        if i % validate_every == 0 and valid is not None:
            logging.info(f"evaluate in the validation set")
            network.eval()

            valid_err = network.calculate_error(valid)
            if valid_err >= valid_best:
                logging.info(f"Worsening on valid: {valid_err} > prev best {valid_best}")
                if c_patience >= patience:
                    logging.info(f"Exceeding patience {c_patience} >= {patience}: STOP")
                    break
                else:
                    c_patience += 1
            else:
                logging.info(f'Found new best model {valid_err}')
                best_parameters = copy.deepcopy(network.state_dict())
                valid_best = valid_err
                c_patience = 0

    sl_end_t: float = perf_counter()
    logging.info(f"Network learning done in {sl_end_t - sl_start_t} secs")
    network.load_state_dict(best_parameters)

    return network, error_history
