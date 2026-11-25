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
    @staticmethod
    def _safe_log_means(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        valid = np.isfinite(arr) & (arr > 0)
        out = np.zeros(arr.shape[0], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            logs = np.where(valid, np.log(np.where(valid, arr, 1.0)), 0.0)
        counts = valid.sum(axis=1)
        pos = counts > 0
        out[pos] = logs[pos].sum(axis=1) / counts[pos]
        return out

    def features(self, units, price, comp, rep_hat) -> np.ndarray:
        """(N, T, 4) : [log ventes, log prix, log prix concurrent, réputation], centrés par produit."""
        u = np.asarray(units, dtype=float)
        p = np.asarray(price, dtype=float)
        c = np.asarray(comp, dtype=float)
        r = np.asarray(rep_hat, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            lu = np.where(np.isfinite(u) & (u > 0), np.log(u) - self.mq[:, None], 0.0)
            lp = np.where(np.isfinite(p) & (p > 0), np.log(p) - self.mp[:, None], 0.0)
            lc = np.where(np.isfinite(c) & (c > 0), np.log(c) - self.mc[:, None], 0.0)
        rep_c = np.where(np.isfinite(r), r - 0.5, 0.0)
        return np.stack([lu, lp, lc, rep_c], axis=2)

    def _xe(self, F, price, w, dlogp=None):
        n = F.shape[0]
        X = F[:, w - self.cfg.window:w, :]
        if dlogp is None:
            pw = np.asarray(price[:, w], dtype=float)
            pw_prev = np.asarray(price[:, w - 1], dtype=float)
            ok = np.isfinite(pw) & (pw > 0) & np.isfinite(pw_prev) & (pw_prev > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                dlogp = np.where(ok, np.log(np.where(ok, pw, 1.0)) - np.log(np.where(ok, pw_prev, 1.0)), 0.0)
        E = np.concatenate([dlogp[:, None], np.repeat(harmonics(w), n, axis=0)], axis=1)
        return X, E

    @staticmethod
    def _flat(X, E):
        return np.concatenate([X.reshape(len(X), -1), E], axis=1)

    def _build_windows(self, world, rep_hat, w_start: int, w_end: int, stat_end: int):
        self.mq = self._safe_log_means(world.units_hist[:, :stat_end])
        self.mp = self._safe_log_means(world.price_hist[:, :stat_end])
        self.mc = self._safe_log_means(world.comp_price[:, :stat_end])
        F = self.features(world.units_hist, world.price_hist, world.comp_price, rep_hat)
        valid_step = (
            np.isfinite(world.units_hist) & (world.units_hist > 0)
            & np.isfinite(world.price_hist) & (world.price_hist > 0)
            & np.isfinite(world.comp_price) & (world.comp_price > 0)
            & np.isfinite(rep_hat)
        )
        Xs, Es, ys = [], [], []
        for w in range(w_start, w_end):
            X, E = self._xe(F, world.price_hist, w)
            win_ok = valid_step[:, w - self.cfg.window:w].all(axis=1) & valid_step[:, w]
            if np.any(win_ok):
                Xs.append(X[win_ok]); Es.append(E[win_ok]); ys.append(F[win_ok, w, 0])
        if not Xs:
            return np.empty((0, self.cfg.window, 4)), np.empty((0, 7)), np.empty((0,))
        return np.concatenate(Xs), np.concatenate(Es), np.concatenate(ys)

    # ---- apprentissage ---------------------------------------------------------------
    def fit(self, world, rep_hat):
        cfg = self.cfg
        t0 = cfg.n_weeks - cfg.test_weeks
        if self.backend == "torch":
            from ..nn_models import train_lstm
            val_weeks = min(cfg.test_weeks, max(4, (t0 - cfg.window) // 5))
            t_val = t0 - val_weeks
            best_epoch = cfg.lstm_epochs
            fallback_model, fallback_stats = None, None
            if t_val > cfg.window:
                X_tr, E_tr, y_tr = self._build_windows(world, rep_hat, cfg.window, t_val, t_val)
                X_va, E_va, y_va = self._build_windows(world, rep_hat, t_val, t0, t_val)
                if len(X_tr) > 0 and len(X_va) > 0:
                    fallback_stats = (self.mq.copy(), self.mp.copy(), self.mc.copy())
                    fallback_model, best_epoch = train_lstm(
                        X_tr, E_tr, y_tr, cfg.seed, epochs=cfg.lstm_epochs,
                        X_val=X_va, E_val=E_va, y_val=y_va, return_best_epoch=True,
                    )
            X, E, y = self._build_windows(world, rep_hat, cfg.window, t0, t0)
            if len(X) > 0:
                self.model = train_lstm(X, E, y, cfg.seed, epochs=best_epoch)
            elif fallback_model is not None:
                self.mq, self.mp, self.mc = fallback_stats
                self.model = fallback_model
        else:
            X, E, y = self._build_windows(world, rep_hat, cfg.window, t0, t0)
            self.model = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                      max_iter=500, random_state=cfg.seed).fit(self._flat(X, E), y)
        self.linear = Ridge(alpha=1.0).fit(self._flat(X, E), y)           # baseline linéaire
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
