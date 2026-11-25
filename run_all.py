"""Pipeline complet : données -> entraînement des agents -> évaluation -> figures -> résultats JSON."""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecom_agents.config import Config
from ecom_agents.data import generate_world
from ecom_agents.evaluation import backtest, elasticity_quality, reputation_quality
from ecom_agents.orchestrator import Orchestrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "light"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--cap", type=float, default=1.05, help="prime max sur le prix concurrent (1.05 = +5 %%)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    cfg = Config(seed=a.seed, output_dir=a.out, shrink=a.shrink, comp_cap_base=a.cap)
    os.makedirs(os.path.join(cfg.output_dir, "figures"), exist_ok=True)

    t = time.time()
    world = generate_world(cfg)
    print(f"[données] {cfg.n_products} produits x {cfg.n_weeks} semaines, {len(world.reviews)} avis ({time.time()-t:.1f}s)")

    orch = Orchestrator(cfg, world, a.backend)
    t = time.time(); orch.fit(); print(f"[entraînement] terminé en {time.time()-t:.1f}s")
    print(f"[backends] demande={orch.demand.model_name} | réputation={orch.reputation.main.backend}")

    results = {"config": {"seed": cfg.seed, "n_products": cfg.n_products, "n_weeks": cfg.n_weeks, "test_weeks": cfg.test_weeks},
               "backends": {"demande": orch.demand.model_name, "reputation": orch.reputation.main.backend},
               "reputation_classifieurs": orch.reputation.metrics, "reputation_qualite": reputation_quality(orch),
               "prevision_demande": orch.demand.forecast_eval(world, orch.rep_hat),
               "segmentation": orch.segmentation.metrics, "elasticite": elasticity_quality(orch),
               "regles_arbre": orch.segmentation.rules, "importances_rf": orch.segmentation.importances}
    bt = backtest(orch, os.path.join(cfg.output_dir, "decisions.jsonl"))
    results["backtest"] = bt["summary"]
    results["backtest_par_categorie"] = bt["by_category"].round(2).to_dict("records")
    with open(os.path.join(cfg.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---------------- figures
    fig_dir = os.path.join(cfg.output_dir, "figures")
    raw = bt["raw"]
    plt.figure(figsize=(6, 4))
    for name, r in raw.items():
        plt.plot(np.cumsum(r["margin_weekly"]), label=name, marker="o", ms=3)
    plt.xlabel("Semaine de test"); plt.ylabel("Marge cumulée"); plt.title("Backtest : marge cumulée par politique")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "backtest_marge.png"), dpi=150); plt.close()

    imp = list(orch.segmentation.importances.items())[:8][::-1]
    plt.figure(figsize=(6, 4)); plt.barh([k for k, _ in imp], [v for _, v in imp])
    plt.title("Random Forest : importance des variables"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "importances_rf.png"), dpi=150); plt.close()

    i = int(np.argmax(world.units_hist[:, :orch.t0].mean(1)))
    t0 = orch.t0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    wk = np.arange(t0, cfg.n_weeks)
    ax[0].plot(wk, raw["multi_agents"]["price_matrix"][i], label="prix multi-agents")
    ax[0].plot(wk, world.comp_price[i, t0:], label="prix concurrent", ls="--")
    ax[0].plot(wk, raw["statique"]["price_matrix"][i], label="prix catalogue", ls=":")
    ax[0].set_title(f"Produit {i} : trajectoire de prix"); ax[0].legend(); ax[0].set_xlabel("semaine")
    ax[1].plot(wk, raw["multi_agents"]["units_matrix"][i], label="ventes multi-agents")
    ax[1].plot(wk, raw["statique"]["units_matrix"][i], label="ventes prix statique", ls=":")
    ax[1].set_title(f"Produit {i} : ventes simulées"); ax[1].legend(); ax[1].set_xlabel("semaine")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "trajectoire_produit.png"), dpi=150); plt.close()

    print(json.dumps({k: v for k, v in results.items() if k not in ("regles_arbre", "importances_rf")}, indent=2, ensure_ascii=False))
    print(results["regles_arbre"])


if __name__ == "__main__":
    main()
