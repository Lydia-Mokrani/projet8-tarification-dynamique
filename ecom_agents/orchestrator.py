"""Orchestrateur multi-agents : état partagé, exécution parallèle, garde-fous, journalisation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .agents.base import Message
from .agents.demand import DemandAgent
from .agents.pricing import PricingAgent
from .agents.reputation import ReputationAgent
from .agents.segmentation import SegmentationAgent
from .agents.vision import VisionAgent
from .config import Config
from .data import World


class State:
    """Vue partagée (tableau noir) : ce que les agents ont observé jusqu'à la semaine w-1."""

    def __init__(self, units, price, comp, rep_hat):
        self.units, self.price, self.comp, self.rep_hat = units, price, comp, rep_hat

    def copy(self) -> "State":
        return State(self.units.copy(), self.price.copy(), self.comp.copy(), self.rep_hat.copy())


class Orchestrator:
    def __init__(self, cfg: Config, world: World, backend: str = "auto"):
        self.cfg, self.world = cfg, world
        self.reputation = ReputationAgent(cfg, backend)
        self.demand = DemandAgent(cfg, backend)
        self.segmentation = SegmentationAgent()
        self.pricing = PricingAgent(cfg)
        self.vision = VisionAgent()
        self.t0 = cfg.n_weeks - cfg.test_weeks

    # ------------------------------------------------------------------ apprentissage
    def fit(self):
        w, cfg = self.world, self.cfg
        self.reputation.fit(w.reviews, self.t0)
        self.rep_hat = self.reputation.build_series(w.reviews, cfg.n_products, cfg.n_weeks)
        self.pricing.fit(w, self.rep_hat)                                  # régression -> élasticités
        self.segmentation.fit(w, self.rep_hat, self.pricing.eps_hat, self.t0, cfg.seed)
        self.pricing.set_segments(self.segmentation.sensitive)
        self.demand.fit(w, self.rep_hat)
        return self

    def initial_state(self) -> State:
        """État au début du backtest : historique connu jusqu'à t0-1, futur masqué."""
        units, price = self.world.units_hist.copy(), self.world.price_hist.copy()
        units[:, self.t0:], price[:, self.t0:] = np.nan, np.nan
        return State(units, price, self.world.comp_price, self.rep_hat)

    # ------------------------------------------------------------------ décision
    def decide(self, w: int, state: State, overrides: dict | None = None):
        """Décide les prix de la semaine w pour tout le catalogue.

        overrides (what-if) : {"comp_mult": {i: 0.85}, "rep_delta": {i: -0.3}} appliqués à la
        dernière semaine observée (w-1)."""
        st = state
        if overrides:
            st = state.copy()
            for i, m in overrides.get("comp_mult", {}).items():
                st.comp[i, w - 1] *= m
            for i, d in overrides.get("rep_delta", {}).items():
                st.rep_hat[i, w - 1] = np.clip(st.rep_hat[i, w - 1] + d, 0, 1)
        with ThreadPoolExecutor(max_workers=3) as ex:                      # agents indépendants en parallèle
            f_d = ex.submit(self.demand.run_batch, st, w)
            f_r = ex.submit(self.reputation.run_batch, st, w)
            f_v = ex.submit(self.vision.run_batch, w)
            q0, rep, vis = f_d.result(), f_r.result(), f_v.result()
        seg = self.segmentation.run_batch()
        p_prev, pc_prev = st.price[:, w - 1], st.comp[:, w - 1]
        prices, info = self.pricing.run_batch(p_prev, pc_prev, rep["rep"], q0, seg["sensitive"])
        # garde-fou final : jamais sous le plancher de marge, arrondi au centime
        floor = self.world.cost * (1 + self.cfg.min_margin)
        prices = np.round(np.maximum(prices, floor), 2)
        decision = {"week": w, "p_prev": p_prev, "pc_prev": pc_prev, "q0": q0, "rep": rep["rep"],
                    "trend": rep["trend"], "sensitive": seg["sensitive"], "proba": seg["proba"],
                    "vision": vis, "info": info, "price": prices}
        return prices, decision

    def messages_for(self, i: int, d: dict) -> list[dict]:
        """Messages JSON échangés pour le produit i (traçabilité / démonstration)."""
        w = d["week"]
        info = d["info"][i]
        return [m.to_dict() for m in (
            Message("demand", i, w, {"ventes_prevues_prix_inchange": d["q0"][i], "modele": self.demand.model_name}),
            Message("reputation", i, w, {"reputation_concurrent": d["rep"][i], "tendance_4sem": d["trend"][i]}),
            Message("vision", i, w, d["vision"], confidence=0.0),
            Message("segmentation", i, w, {"sensible_au_prix": bool(d["sensitive"][i]), "proba": d["proba"][i]}, confidence=float(max(d["proba"][i], 1 - d["proba"][i]))),
            Message("pricing", i, w, {"prix_precedent": d["p_prev"][i], "prix_recommande": d["price"][i], "elasticite": info["eps"],
                                      "prix_optimal_non_contraint": info["p_star_non_contraint"], "contrainte_active": info["reason"]}),
        )]
