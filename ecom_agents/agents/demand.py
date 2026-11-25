"""Agent Demande : prévision hebdomadaire des ventes par LSTM (repli : MLP scikit-learn)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from ..config import Config, resolve_backend
from .base import Agent


def harmonics(w) -> np.ndarray:
    """Saisonnalité annuelle : 3 harmoniques (sin, cos) -> (len(w), 6)."""
    w = np.atleast_1d(w).astype(float)
    return np.stack([f(2 * np.pi * k * w / 52) for k in (1, 2, 3) for f in (np.sin, np.cos)], axis=1)


class DemandAgent(Agent):
    name = "demand"

    def __init__(self, cfg: Config, backend: str = "auto"):
        self.cfg = cfg
        self.backend = resolve_backend(backend, "lstm")
        self.model_name = "LSTM" if self.backend == "torch" else "MLP (repli)"
        self.model = None
        self.linear = None

    # ---- construction des entrées -------------------------------------------------
    def features(self, units, price, comp, rep_hat) -> np.ndarray:
        """(N, T, 4) : [log ventes, log prix, log prix concurrent, réputation], centrés par produit."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.stack([np.log(units) - self.mq[:, None], np.log(price) - self.mp[:, None],
                             np.log(comp) - self.mc[:, None], rep_hat - 0.5], axis=2)

    def _xe(self, F, price, w, dlogp=None):
        n = F.shape[0]
        X = F[:, w - self.cfg.window:w, :]
        if dlogp is None:
            dlogp = np.log(price[:, w]) - np.log(price[:, w - 1])
        E = np.concatenate([dlogp[:, None], np.repeat(harmonics(w), n, axis=0)], axis=1)
        return X, E

    @staticmethod
    def _flat(X, E):
        return np.concatenate([X.reshape(len(X), -1), E], axis=1)

    # ---- apprentissage ---------------------------------------------------------------
    def fit(self, world, rep_hat):
        cfg = self.cfg
        t0 = cfg.n_weeks - cfg.test_weeks
        self.mq = np.log(world.units_hist[:, :t0]).mean(1)
        self.mp = np.log(world.price_hist[:, :t0]).mean(1)
        self.mc = np.log(world.comp_price[:, :t0]).mean(1)
        F = self.features(world.units_hist, world.price_hist, world.comp_price, rep_hat)
        Xs, Es, ys = [], [], []
        for w in range(cfg.window, t0):
            X, E = self._xe(F, world.price_hist, w)
            Xs.append(X); Es.append(E); ys.append(F[:, w, 0])
        X, E, y = np.concatenate(Xs), np.concatenate(Es), np.concatenate(ys)
        self.linear = Ridge(alpha=1.0).fit(self._flat(X, E), y)           # baseline linéaire
        if self.backend == "torch":
            from ..nn_models import train_lstm
            self.model = train_lstm(X, E, y, cfg.seed, epochs=cfg.lstm_epochs)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                      max_iter=500, random_state=cfg.seed).fit(self._flat(X, E), y)
        return self

    # ---- prédiction ---------------------------------------------------------------------
    def predict_centered(self, X, E) -> np.ndarray:
        if self.backend == "torch":
            from ..nn_models import predict_lstm
            return predict_lstm(self.model, X, E)
        return self.model.predict(self._flat(X, E))

    def run_batch(self, state, w: int) -> np.ndarray:
        """Ventes prévues pour la semaine w *si le prix reste inchangé* (variation de prix = 0)."""
        F = self.features(state.units, state.price, state.comp, state.rep_hat)
        X, E = self._xe(F, state.price, w, dlogp=np.zeros(F.shape[0]))
        return np.exp(self.predict_centered(X, E) + self.mq)

    def forecast_eval(self, world, rep_hat) -> dict:
        """MAPE sur la période de test (prix réellement pratiqués connus) : modèle vs baselines."""
        cfg = self.cfg
        t0 = cfg.n_weeks - cfg.test_weeks
        F = self.features(world.units_hist, world.price_hist, world.comp_price, rep_hat)
        res = {"model": [], "ridge": [], "naive": []}
        for w in range(t0, cfg.n_weeks):
            X, E = self._xe(F, world.price_hist, w)
            actual = world.units_hist[:, w]
            res["model"].append(np.abs(np.exp(self.predict_centered(X, E) + self.mq) - actual) / actual)
            res["ridge"].append(np.abs(np.exp(self.linear.predict(self._flat(X, E)) + self.mq) - actual) / actual)
            res["naive"].append(np.abs(world.units_hist[:, w - 1] - actual) / actual)
        return {f"MAPE_{self.model_name if k == 'model' else k}": float(np.mean(v)) for k, v in res.items()}
