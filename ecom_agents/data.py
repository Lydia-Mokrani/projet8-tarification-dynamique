"""Données du projet.

Le sujet impose deux jeux Kaggle (E-Commerce Sales Forecast + Amazon/Flipkart
Product Reviews). Comme ils ne sont pas téléchargeables automatiquement ici, ce
module fournit :

1. `generate_world` : un monde synthétique dont la structure imite ces jeux
   (ventes hebdomadaires, prix, prix/réputation concurrents, avis texte+note).
   Les paramètres "vrais" (élasticités...) sont connus, ce qui permet un
   *backtest* honnête de la politique de prix (impossible avec des données
   réelles, où le contrefactuel n'est pas observé).
2. `world_from_tables` : un adaptateur pour brancher les vraies données Kaggle
   une fois mises au format tabulaire attendu (voir README).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

# Élasticité-prix moyenne de la demande par catégorie
CATEGORIES = {"electronique": -2.4, "mode": -2.0, "beaute": -1.8, "maison": -1.5, "epicerie": -1.1}
KAPPA = 0.8  # effet de la réputation concurrente sur notre demande

POS = ["excellent", "great", "love", "perfect", "fast", "durable", "worth", "recommend",
       "sturdy", "amazing", "superb", "reliable", "smooth", "impressive"]
NEG = ["terrible", "broke", "poor", "slow", "waste", "disappointed", "refund", "cheap",
       "awful", "defective", "useless", "faulty", "overpriced", "flimsy"]
FILL = ["product", "delivery", "price", "quality", "arrived", "battery", "screen", "packaging",
        "size", "color", "seller", "order", "item", "after", "week", "really", "very", "bought",
        "using", "service", "design", "material", "support", "box"]


def season_factor(w):
    """Saisonnalité annuelle + pic de fin d'année (semaine ~47)."""
    w = np.asarray(w, dtype=float)
    return 1 + 0.25 * np.sin(2 * np.pi * (w - 10) / 52) + 0.35 * np.exp(-0.5 * (((w % 52) - 47) / 2.5) ** 2)


@dataclass
class World:
    cfg: Config
    products: pd.DataFrame
    p_ref: np.ndarray        # prix catalogue
    cost: np.ndarray         # coût unitaire
    units_hist: np.ndarray   # (N, T) ventes observées sous prix historiques
    price_hist: np.ndarray   # (N, T) prix historiques (avec promotions aléatoires)
    comp_price: np.ndarray   # (N, T) prix du concurrent équivalent
    reviews: pd.DataFrame    # product_id, week, text, rating (avis sur le concurrent)
    # --- paramètres vrais (None pour des données réelles) ---
    base_q: np.ndarray | None = None
    eps_true: np.ndarray | None = None
    gamma_true: np.ndarray | None = None
    comp_ref: np.ndarray | None = None
    rep_true: np.ndarray | None = None
    noise: np.ndarray | None = None
    capacity: np.ndarray | None = None   # stock disponible par semaine (unités)
    simulatable: bool = True

    def sales(self, w: int, p: np.ndarray) -> np.ndarray:
        """Ventes réalisées = demande vraie plafonnée par le stock."""
        return np.minimum(self.demand(w, p), self.capacity)

    def demand_i(self, i: int, w: int, p):
        """Demande vraie du produit i, semaine w, pour un prix ou un vecteur de prix p."""
        return (self.base_q[i] * season_factor(w) * (p / self.p_ref[i]) ** self.eps_true[i]
                * (self.comp_price[i, w] / self.comp_ref[i]) ** self.gamma_true[i]
                * np.exp(-KAPPA * (self.rep_true[i, w] - 0.5)) * np.exp(self.noise[i, w]))

    def demand(self, w: int, p: np.ndarray) -> np.ndarray:
        """Demande vraie de la semaine w pour le vecteur de prix p (environnement)."""
        return (self.base_q * season_factor(w) * (p / self.p_ref) ** self.eps_true
                * (self.comp_price[:, w] / self.comp_ref) ** self.gamma_true
                * np.exp(-KAPPA * (self.rep_true[:, w] - 0.5)) * np.exp(self.noise[:, w]))


def _make_review(rng: np.random.Generator, polarity: int) -> str:
    if polarity == 0:
        words = [str(rng.choice(POS)), str(rng.choice(NEG))]
    else:
        main, other = (POS, NEG) if polarity > 0 else (NEG, POS)
        words = [str(x) for x in rng.choice(main, rng.integers(2, 4), replace=False)]
        if rng.random() < 0.30:           # avis nuancé : un mot de polarité opposée
            words.append(str(rng.choice(other)))
    fill = [str(x) for x in rng.choice(FILL, rng.integers(3, 6), replace=False)]
    toks = words + fill
    rng.shuffle(toks)
    return " ".join(toks)


def generate_world(cfg: Config) -> World:
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_products, cfg.n_weeks

    cats = rng.choice(list(CATEGORIES), N)
    eps = np.array([CATEGORIES[c] for c in cats]) + rng.normal(0, 0.25, N)
    p_ref = np.round(np.exp(rng.normal(3.6, 0.7, N)), 2)
    cost = p_ref * rng.uniform(0.45, 0.72, N)
    base_q = np.exp(rng.normal(4.5, 0.5, N))
    gamma = rng.uniform(0.3, 0.9, N)
    comp_ref = p_ref * rng.uniform(0.92, 1.08, N)

    # prix concurrent : AR(1) en log + chocs (soldes / hausses)
    log_pc = np.zeros((N, T))
    for w in range(1, T):
        shock = (rng.random(N) < 0.05) * rng.choice([-0.12, 0.08], N)
        log_pc[:, w] = 0.9 * log_pc[:, w - 1] + rng.normal(0, 0.03, N) + shock
    comp = comp_ref[:, None] * np.exp(log_pc)

    # réputation concurrente vraie (probabilité qu'un avis soit positif)
    rep = np.zeros((N, T))
    rep[:, 0] = rng.uniform(0.35, 0.8, N)
    for w in range(1, T):
        shock = (rng.random(N) < 0.04) * rng.choice([-0.15, 0.12], N)
        rep[:, w] = np.clip(rep[:, w - 1] + 0.1 * (0.6 - rep[:, w - 1]) + rng.normal(0, 0.025, N) + shock, 0.15, 0.9)

    # historique de prix avec promotions aléatoires (nécessaire pour identifier l'élasticité)
    levels = np.zeros((N, T))
    cur = np.zeros(N)
    steps = np.array([-0.15, -0.10, -0.05, 0, 0, 0.05, 0.10])
    for w in range(T):
        change = rng.random(N) < 0.3
        cur = np.where(change, rng.choice(steps, N), cur)
        levels[:, w] = cur
    price_hist = np.round(p_ref[:, None] * (1 + levels), 2)
    noise = rng.normal(0, 0.08, (N, T))

    products = pd.DataFrame({"product_id": np.arange(N), "category": cats, "p_ref": p_ref, "cost": np.round(cost, 2)})
    world = World(cfg=cfg, products=products, p_ref=p_ref, cost=cost,
                  units_hist=np.zeros((N, T)), price_hist=price_hist, comp_price=comp,
                  reviews=pd.DataFrame(), base_q=base_q, eps_true=eps, gamma_true=gamma,
                  comp_ref=comp_ref, rep_true=rep, noise=noise,
                  capacity=1.10 * base_q)
    for w in range(T):
        world.units_hist[:, w] = world.demand(w, price_hist[:, w])

    # avis clients du concurrent (texte + note)
    rows_pid, rows_w, rows_txt, rows_rating = [], [], [], []
    for i in range(N):
        for w in range(T):
            for _ in range(max(1, rng.poisson(cfg.reviews_per_week))):
                u = rng.random()
                pol = 0 if u < 0.08 else (1 if u < 0.08 + 0.92 * rep[i, w] else -1)
                txt = _make_review(rng, pol)
                if pol == 0:
                    rating = 3
                else:
                    rating = int(rng.choice([4, 5] if pol > 0 else [1, 2]))
                    if rng.random() < 0.05:   # bruit d'étiquetage (note incohérente avec le texte)
                        rating = 6 - rating
                rows_pid.append(i); rows_w.append(w); rows_txt.append(txt); rows_rating.append(rating)
    world.reviews = pd.DataFrame({"product_id": rows_pid, "week": rows_w, "text": rows_txt, "rating": rows_rating})
    return world


def world_from_tables(cfg: Config, products: pd.DataFrame, weekly: pd.DataFrame, reviews: pd.DataFrame) -> World:
    """Adaptateur pour de vraies données (non simulables : pas de backtest contrefactuel).

    products : product_id, category, p_ref, cost
    weekly   : product_id, week, units, price, comp_price
    reviews  : product_id, week, text, rating
    """
    for df, cols in ((products, ["product_id", "category", "p_ref", "cost"]),
                     (weekly, ["product_id", "week", "units", "price", "comp_price"]),
                     (reviews, ["product_id", "week", "text", "rating"])):
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    products = products.sort_values("product_id").reset_index(drop=True)
    ids = products["product_id"].to_numpy()
    idx = {pid: k for k, pid in enumerate(ids)}
    N, T = len(ids), int(weekly["week"].max()) + 1
    mats = {c: np.full((N, T), np.nan) for c in ("units", "price", "comp_price")}
    for r in weekly.itertuples(index=False):
        k = idx[r.product_id]
        for c in mats:
            mats[c][k, int(r.week)] = getattr(r, c)
    rev = reviews.copy()
    rev["product_id"] = rev["product_id"].map(idx)
    cfg.n_products, cfg.n_weeks = N, T
    return World(cfg=cfg, products=products, p_ref=products["p_ref"].to_numpy(float),
                 cost=products["cost"].to_numpy(float), units_hist=mats["units"],
                 price_hist=mats["price"], comp_price=mats["comp_price"], reviews=rev,
                 simulatable=False)
