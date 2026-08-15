from __future__ import annotations

import flwr as fl
import torch

from compression import compress_gradient, flatten_tensors
from model import get_model
from train_federated import get_parameters, set_parameters, train_one_epoch


class PathologyClient(fl.client.NumPyClient):
    def __init__(self, trainloader, model_name: str = "tiny", k_ratio: float = 0.1):
        self.trainloader = trainloader
        self.model = get_model(model_name)
        self.k_ratio = k_ratio

    def get_parameters(self, config):
        return [tensor.numpy() for tensor in get_parameters(self.model)]

    def fit(self, parameters, config):
        old_parameters = [torch.tensor(parameter) for parameter in parameters]
        set_parameters(self.model, old_parameters)
        train_one_epoch(self.model, self.trainloader, torch.device("cpu"), lr=float(config.get("lr", 1e-3)))
        new_parameters = get_parameters(self.model)

        updates = [new - old for new, old in zip(new_parameters, old_parameters)]
        flat_update, _ = flatten_tensors(updates)
        compressed = compress_gradient(flat_update, k_ratio=self.k_ratio, use_topk=True, use_sign=True)

        return [compressed.numpy()], len(self.trainloader.dataset), {"compression": "topk_sign"}
