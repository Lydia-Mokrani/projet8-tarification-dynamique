"""Modèles PyTorch : LSTM de prévision de la demande et CNN d'attributs visuels.

Ce module n'est importé que si torch est installé (voir config.has_torch).
"""
from __future__ import annotations

import copy

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


def train_lstm(
    X,
    E,
    y,
    seed: int,
    epochs: int = 60,
    lr: float = 3e-3,
    batch_size: int = 128,
    hidden: int = 32,
    X_val=None,
    E_val=None,
    y_val=None,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 10,
    return_best_epoch: bool = False,
):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Et = torch.tensor(E, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = LSTMNet(X.shape[2], E.shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    n = len(Xt)

    use_es = X_val is not None and E_val is not None and y_val is not None and len(X_val) > 0
    if use_es:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        Ev = torch.tensor(E_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)
        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = epochs
        no_imp = 0
    else:
        sched = None
        best_epoch = epochs

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx], Et[idx]), yt[idx])
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
        if use_es:
            net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(net(Xv, Ev), yv).item())
            sched.step(vloss)
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = ep
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break
    if use_es:
        net.load_state_dict(best_state)
    net.eval()
    return (net, best_epoch) if return_best_epoch else net


def predict_lstm(net, X, E) -> np.ndarray:
    net.eval()
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
