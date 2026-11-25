"""Agent Pricing : régression linéaire (élasticité) + optimisation de marge sous contraintes."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from ..config import Config
from .base import Agent
from .demand import harmonics


class PricingAgent(Agent):
    name = "pricing"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, world, rep_hat):
        """Régression log-log par produit : log q = a + eps*log p + gamma*log p_conc + d*rep + saison."""
        t0 = self.cfg.n_weeks - self.cfg.test_weeks
        self.cost, self.p_ref = world.cost, world.p_ref
        self.capacity = world.capacity if world.capacity is not None else np.full(len(world.cost), np.inf)
        H = harmonics(np.arange(t0))
        n = len(world.p_ref)
        self.eps_hat, self.gamma_hat = np.zeros(n), np.zeros(n)
        for i in range(n):
            X = np.column_stack([np.log(world.price_hist[i, :t0]), np.log(world.comp_price[i, :t0]), rep_hat[i, :t0], H])
            reg = LinearRegression().fit(X, np.log(world.units_hist[i, :t0]))
            self.eps_hat[i], self.gamma_hat[i] = reg.coef_[0], reg.coef_[1]
        self.eps_hat = np.clip(self.eps_hat, -4.0, -0.3)
        return self

    def set_segments(self, sensitive: np.ndarray):
        """Rétrécissement (shrinkage) de l'élasticité vers la moyenne du segment prédit."""
        self.sensitive = sensitive
        self.seg_eps = {True: float(self.eps_hat[sensitive].mean()), False: float(self.eps_hat[~sensitive].mean())}
        prior = np.where(sensitive, self.seg_eps[True], self.seg_eps[False])
        self.eps = (1 - self.cfg.shrink) * self.eps_hat + self.cfg.shrink * prior

    def run_batch(self, p_prev, pc_prev, rep, q0, sensitive, overrides_eps=None):
        cfg = self.cfg
        n = len(p_prev)
        prices, info = np.zeros(n), []
        for i in range(n):
            step = cfg.step_sensitive if sensitive[i] else cfg.step_insensitive
            cap = pc_prev[i] * (cfg.comp_cap_base + 0.3 * (0.5 - rep[i]))     # plafond lié au concurrent et à sa réputation
            bounds_hi = {"variation_max": p_prev[i] * (1 + step), "borne_catalogue": self.p_ref[i] * cfg.list_bounds[1], "plafond_concurrent": cap}
            bounds_lo = {"variation_max": p_prev[i] * (1 - step), "plancher_marge": self.cost[i] * (1 + cfg.min_margin), "borne_catalogue": self.p_ref[i] * cfg.list_bounds[0]}
            hi_name, hi = min(bounds_hi.items(), key=lambda kv: kv[1])
            lo_name, lo = max(bounds_lo.items(), key=lambda kv: kv[1])
            eps = self.eps[i]
            if hi < lo:                                          # contraintes incompatibles : on protège la marge
                p, reason = lo, f"plancher ({lo_name}) prioritaire"
            else:
                grid = np.linspace(lo, hi, 41)
                profit = (grid - self.cost[i]) * np.minimum(q0[i] * (grid / p_prev[i]) ** eps, self.capacity[i])
                k = int(np.argmax(profit))
                p = grid[k]
                reason = f"borne haute ({hi_name})" if k == len(grid) - 1 else (f"borne basse ({lo_name})" if k == 0 else "optimum intérieur")
            prices[i] = p
            p_star = self.cost[i] * eps / (1 + eps) if eps < -1 else np.nan
            info.append({"eps": float(eps), "p_star_non_contraint": float(p_star), "lo": float(lo), "hi": float(hi), "reason": reason})
        return prices, info
