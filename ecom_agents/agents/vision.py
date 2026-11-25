"""Agent Vision (CNN) : attributs visuels des produits. Optionnel : nécessite des images."""
from __future__ import annotations

import numpy as np

from ..config import has_torch
from .base import Agent


class VisionAgent(Agent):
    name = "vision"

    def __init__(self):
        self.enabled = False
        self.net = None

    def fit(self, images: np.ndarray, labels: np.ndarray, n_classes: int, seed: int = 0, epochs: int = 10):
        """images : (N, 3, H, W) dans [0, 1] ; labels : classe de gamme / qualité perçue."""
        if not has_torch():
            raise RuntimeError("Le CNN nécessite PyTorch.")
        from ..nn_models import train_cnn
        self.net = train_cnn(images, labels, n_classes, seed, epochs)
        self.enabled = True
        return self

    def predict_tier(self, images: np.ndarray) -> np.ndarray:
        from ..nn_models import predict_cnn
        return predict_cnn(self.net, images)

    def run_batch(self, w: int) -> dict:
        # Sans images (cas du jeu de données du sujet), l'agent s'abstient : aucun signal visuel.
        return {"enabled": self.enabled}
