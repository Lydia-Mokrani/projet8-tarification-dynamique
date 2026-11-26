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
        self.eps_hat, self.gamma_hat = np.full(n, -1.5), np.zeros(n)
        for i in range(n):
            p_i = np.asarray(world.price_hist[i, :t0], dtype=float)
            c_i = np.asarray(world.comp_price[i, :t0], dtype=float)
            q_i = np.asarray(world.units_hist[i, :t0], dtype=float)
            r_i = np.asarray(rep_hat[i, :t0], dtype=float)
            valid = (
                np.isfinite(p_i) & (p_i > 0)
                & np.isfinite(c_i) & (c_i > 0)
                & np.isfinite(q_i) & (q_i > 0)
                & np.isfinite(r_i)
            )
            if valid.sum() >= 30:
                X = np.column_stack([np.log(p_i[valid]), np.log(c_i[valid]), r_i[valid], H[valid]])
                reg = LinearRegression().fit(X, np.log(q_i[valid]))
                self.eps_hat[i], self.gamma_hat[i] = reg.coef_[0], reg.coef_[1]
        self.eps_hat = np.clip(self.eps_hat, -4.0, -0.3)
        return self

    def set_segments(self, sensitive: np.ndarray):
        """Rétrécissement (shrinkage) de l'élasticité vers la moyenne du segment prédit."""
        sensitive = np.asarray(sensitive, dtype=bool)
        self.sensitive = sensitive
        finite = np.isfinite(self.eps_hat)
        fallback = float(np.mean(self.eps_hat[finite])) if np.any(finite) else -1.5
        self.seg_eps = {}
        for flag, mask in ((True, sensitive), (False, ~sensitive)):
            sel = mask & finite
            self.seg_eps[flag] = float(np.mean(self.eps_hat[sel])) if np.any(sel) else fallback
        prior = np.where(sensitive, self.seg_eps[True], self.seg_eps[False])
        self.eps = (1 - self.cfg.shrink) * self.eps_hat + self.cfg.shrink * prior

    def run_batch(self, p_prev, pc_prev, rep, q0, sensitive, overrides_eps=None):
        cfg = self.cfg
        n = len(p_prev)
        prices, info = np.zeros(n), []
        is_dict_ovr = isinstance(overrides_eps, dict)
        for i in range(n):
            p_prev_i = float(p_prev[i]) if np.isfinite(p_prev[i]) and p_prev[i] > 0 else float(self.p_ref[i])
            rep_i = float(rep[i]) if np.isfinite(rep[i]) else 0.5
            q0_i = float(q0[i]) if np.isfinite(q0[i]) and q0[i] >= 0 else 0.0
            step = cfg.step_sensitive if sensitive[i] else cfg.step_insensitive
            floor_i = float(np.ceil(self.cost[i] * (1 + cfg.min_margin) * 100.0 - 1e-9) / 100.0)
            bounds_hi = {"variation_max": p_prev_i * (1 + step), "borne_catalogue": self.p_ref[i] * cfg.list_bounds[1]}
            if np.isfinite(pc_prev[i]) and pc_prev[i] > 0:
                bounds_hi["plafond_concurrent"] = float(pc_prev[i]) * (cfg.comp_cap_base + 0.3 * (0.5 - rep_i))
            bounds_lo = {"variation_max": p_prev_i * (1 - step), "plancher_marge": floor_i, "borne_catalogue": self.p_ref[i] * cfg.list_bounds[0]}
            hi_name, hi = min(bounds_hi.items(), key=lambda kv: kv[1])
            lo_name, lo = max(bounds_lo.items(), key=lambda kv: kv[1])
            eps = self.eps[i]
            if overrides_eps is not None:
                ovr = overrides_eps.get(i) if is_dict_ovr else overrides_eps[i]
                if ovr is not None and np.isfinite(ovr):
                    eps = float(np.clip(ovr, -4.0, -0.3))
            if hi < lo:                                          # contraintes incompatibles : la borne basse l'emporte
                p, reason = lo, f"borne basse ({lo_name}) prioritaire sur {hi_name}"
            else:
                grid = np.linspace(lo, hi, 41)
                profit = (grid - self.cost[i]) * np.minimum(q0_i * (grid / p_prev_i) ** eps, self.capacity[i])
                k = int(np.argmax(profit))
                p = grid[k]
                reason = f"borne haute ({hi_name})" if k == len(grid) - 1 else (f"borne basse ({lo_name})" if k == 0 else "optimum intérieur")
            prices[i] = p
            p_star = self.cost[i] * eps / (1 + eps) if eps < -1 else np.nan
            info.append({"eps": float(eps), "p_star_non_contraint": float(p_star), "lo": float(lo), "hi": float(hi), "reason": reason})
        return prices, info
