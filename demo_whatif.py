"""Démonstration : réaction des agents à des événements (baisse de prix concurrent, vague d'avis négatifs)."""
import argparse, json
import numpy as np
from ecom_agents.config import Config
from ecom_agents.data import generate_world
from ecom_agents.orchestrator import Orchestrator

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="auto"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--product", type=int, default=-1, help="id produit (défaut : plus gros volume)")
ap.add_argument("--weeks_ahead", type=int, default=3, help="nombre de semaines simulées avant l'événement")
ap.add_argument("--cap", type=float, default=1.05)
a = ap.parse_args()

cfg = Config(seed=a.seed, comp_cap_base=a.cap)
world = generate_world(cfg)
orch = Orchestrator(cfg, world, a.backend).fit()
state = orch.initial_state()
i = a.product if a.product >= 0 else int(np.argmax(world.units_hist[:, :orch.t0].mean(1)))
for w in range(orch.t0, orch.t0 + a.weeks_ahead):                       # on fait tourner le système quelques semaines
    p, _ = orch.decide(w, state)
    state.units[:, w], state.price[:, w] = world.demand(w, p), p
w = orch.t0 + a.weeks_ahead

scenarios = {"0. situation normale": None,
             "1. concurrent baisse son prix de 15 %": {"comp_mult": {i: 0.85}},
             "2. concurrent baisse son prix de 15 % ET avis +0.30 de positif": {"comp_mult": {i: 0.85}, "rep_delta": {i: +0.30}},
             "3. vague d'avis négatifs sur le concurrent (-0.30)": {"rep_delta": {i: -0.30}}}
cat = world.products.loc[i, "category"]
print(f"Produit {i} ({cat}) | semaine {w} | coût {world.cost[i]:.2f} | prix précédent {state.price[i, w-1]:.2f} | prix concurrent {world.comp_price[i, w-1]:.2f}\n")
for name, ov in scenarios.items():
    prices, dec = orch.decide(w, state, ov)
    m = orch.messages_for(i, dec)
    pr = m[-1]["payload"]
    print(f"=== {name}\n    réputation concurrent={dec['rep'][i]:.2f} | demande prévue (prix inchangé)={dec['q0'][i]:.1f} | "
          f"segment={'sensible' if dec['sensitive'][i] else 'peu sensible'} | ε={pr['elasticite']:.2f}\n"
          f"    -> prix recommandé {pr['prix_recommande']:.2f} (précédent {pr['prix_precedent']:.2f}) ; contrainte active : {pr['contrainte_active']}")
print("\nExemple de messages JSON échangés (scénario 0) :")
_, dec = orch.decide(w, state)
print(json.dumps(orch.messages_for(i, dec), indent=1, ensure_ascii=False, default=float))
