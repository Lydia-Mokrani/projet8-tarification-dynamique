"""Évaluation : prévision (MAPE), classification (précision/F1) et backtest des politiques de prix."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .orchestrator import Orchestrator


def run_policy(orch: Orchestrator, policy: str, log_path: str | None = None) -> dict:
    """Simule les semaines de test avec l'environnement vrai (demande vraie + bruit commun)."""
    world, cfg = orch.world, orch.cfg
    t0, t = orch.t0, cfg.n_weeks
    state = orch.initial_state()
    lo_b, hi_b = cfg.list_bounds
    floor = world.cost * (1 + cfg.min_margin)
    margin_w, revenue_w, units_w, price_traj = [], [], [], []
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    for w in range(t0, t):
        if policy == "statique":
            p = world.p_ref.copy()
        elif policy == "alignement_concurrent":
            p = np.clip(world.comp_price[:, w - 1] * 0.98, np.maximum(floor, world.p_ref * lo_b), world.p_ref * hi_b)
        elif policy == "multi_agents":
            p, dec = orch.decide(w, state)
            if log:
                for i in range(cfg.n_products):
                    log.write(json.dumps({"week": w, "product_id": i, "messages": orch.messages_for(i, dec)}, ensure_ascii=False, default=float) + "\n")
        elif policy == "oracle":
            # borne supérieure à un pas : connaît la vraie demande (bruit compris), mêmes contraintes
            p_prev = state.price[:, w - 1]
            p = np.zeros(cfg.n_products)
            for i in range(cfg.n_products):
                step = cfg.step_sensitive if orch.segmentation.sensitive[i] else cfg.step_insensitive
                cap = world.comp_price[i, w - 1] * (cfg.comp_cap_base + 0.3 * (0.5 - orch.rep_hat[i, w - 1]))
                hi = min(p_prev[i] * (1 + step), world.p_ref[i] * hi_b, cap)
                lo = max(p_prev[i] * (1 - step), floor[i], world.p_ref[i] * lo_b)
                if hi < lo:
                    p[i] = lo
                    continue
                g = np.linspace(lo, hi, 81)
                p[i] = g[int(np.argmax((g - world.cost[i]) * np.minimum(world.demand_i(i, w, g), world.capacity[i])))]
            p = np.round(p, 2)
        else:
            raise ValueError(policy)
        demand = world.demand(w, p)
        q = np.minimum(demand, world.capacity)                      # ventes réalisées (stock limité)
        state.units[:, w], state.price[:, w] = demand, p           # les agents observent la demande latente
        margin_w.append(float(((p - world.cost) * q).sum()))
        revenue_w.append(float((p * q).sum()))
        units_w.append(float(q.sum()))
        price_traj.append(p.copy())
    if log:
        log.close()
    sold = np.minimum(state.units[:, t0:], world.capacity[:, None])
    return {"margin_weekly": margin_w, "margin": sum(margin_w), "revenue": sum(revenue_w), "units": sum(units_w),
            "prices": np.array(price_traj), "units_matrix": sold, "price_matrix": state.price[:, t0:]}


def backtest(orch: Orchestrator, log_path: str | None = None) -> dict:
    out = {p: run_policy(orch, p, log_path if p == "multi_agents" else None)
           for p in ("statique", "alignement_concurrent", "multi_agents", "oracle")}
    base = out["statique"]["margin"]
    summary = {p: {"marge_totale": r["margin"], "chiffre_affaires": r["revenue"], "unites": r["units"],
                   "gain_marge_vs_statique_%": 100 * (r["margin"] / base - 1)} for p, r in out.items()}
    # détail par catégorie
    cats = orch.world.products["category"].to_numpy()
    rows = []
    for c in sorted(set(cats)):
        m = cats == c
        def mg(pol):
            r = out[pol]
            return float(((r["price_matrix"][m] - orch.world.cost[m, None]) * r["units_matrix"][m]).sum())
        rows.append({"categorie": c, "n": int(m.sum()), "gain_%_multi_agents": 100 * (mg("multi_agents") / mg("statique") - 1),
                     "gain_%_alignement": 100 * (mg("alignement_concurrent") / mg("statique") - 1)})
    return {"summary": summary, "by_category": pd.DataFrame(rows), "raw": out}


def reputation_quality(orch: Orchestrator) -> dict:
    """Le score de réputation estimé par les avis suit-il la vraie réputation (inconnue des agents) ?"""
    w = orch.world
    return {"correlation_rep_estimee_vs_vraie": float(np.corrcoef(orch.rep_hat.ravel(), w.rep_true.ravel())[0, 1])}


def elasticity_quality(orch: Orchestrator) -> dict:
    w = orch.world
    return {"elasticite_MAE_regression": float(np.mean(np.abs(orch.pricing.eps_hat - w.eps_true))),
            "elasticite_MAE_apres_shrinkage": float(np.mean(np.abs(orch.pricing.eps - w.eps_true))),
            "elasticite_correlation": float(np.corrcoef(orch.pricing.eps, w.eps_true)[0, 1])}
