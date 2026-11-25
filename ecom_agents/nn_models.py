"""Modèles PyTorch : LSTM de prévision de la demande et CNN d'attributs visuels.

Ce module n'est importé que si torch est installé (voir config.has_torch).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """LSTM sur la fenêtre historique + covariables connues de la semaine cible.

    x : (batch, window, d_in)  historique [log ventes, log prix, log prix concurrent, réputation]
    e : (batch, d_extra)       covariables de la semaine à prévoir [variation de prix, saisonnalité]
    """

    def __init__(self, d_in: int, d_extra: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + d_extra, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, e):
        _, (hn, _) = self.lstm(x)
        return self.head(torch.cat([hn[-1], e], dim=1)).squeeze(1)


def train_lstm(X, E, y, seed: int, epochs: int = 60, lr: float = 3e-3, batch_size: int = 128, hidden: int = 32):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Et = torch.tensor(E, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = LSTMNet(X.shape[2], E.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(Xt)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx], Et[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    return net


def predict_lstm(net, X, E) -> np.ndarray:
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32), torch.tensor(E, dtype=torch.float32)).numpy()


class CNNNet(nn.Module):
    """Petit CNN pour classer des images produit (ex. niveau de gamme / qualité perçue)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(images, labels, n_classes: int, seed: int, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32):
    """images : (N, 3, H, W) float32 dans [0, 1] ; labels : (N,) entiers."""
    torch.manual_seed(seed)
    Xt = torch.tensor(images, dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.long)
    net = CNNNet(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    return net


def predict_cnn(net, images) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(images, dtype=torch.float32)), dim=1).numpy()
