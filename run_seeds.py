"""Robustesse : répète le backtest sur plusieurs graines et plusieurs variantes (ablations)."""
import argparse, json
import numpy as np
from ecom_agents.config import Config
from ecom_agents.data import generate_world
from ecom_agents.evaluation import backtest, elasticity_quality
from ecom_agents.orchestrator import Orchestrator

ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="auto"); ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
VARIANTS = {"base (shrink=0, plafond=1.05)": {}, "shrink=0.5": {"shrink": 0.5}, "plafond concurrent=1.15": {"comp_cap_base": 1.15}}
out = {}
for name, kw in VARIANTS.items():
    g = {"multi_agents": [], "alignement_concurrent": [], "oracle": []}; mae = []
    for seed in range(1, a.seeds + 1):
        cfg = Config(seed=seed, **kw)
        orch = Orchestrator(cfg, generate_world(cfg), a.backend).fit()
        bt = backtest(orch)["summary"]
        for k in g:
            g[k].append(bt[k]["gain_marge_vs_statique_%"])
        mae.append(elasticity_quality(orch)["elasticite_MAE_apres_shrinkage"])
    out[name] = {**{f"gain_{k}_moyen_%": float(np.mean(v)) for k, v in g.items()},
                 "ecart_type_multi_agents": float(np.std(g["multi_agents"])),
                 "multi_agents_par_graine": [round(x, 2) for x in g["multi_agents"]],
                 "MAE_elasticite": float(np.mean(mae))}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
json.dump(out, open("outputs/robustesse_graines.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
