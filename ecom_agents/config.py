"""Configuration globale du projet + détection des bibliothèques disponibles."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_products: int = 60          # nombre de produits du catalogue simulé
    n_weeks: int = 104            # 2 ans d'historique hebdomadaire
    test_weeks: int = 12          # dernières semaines réservées au backtest
    window: int = 12              # fenêtre d'entrée du LSTM (semaines)
    reviews_per_week: int = 6     # avis concurrents moyens par produit et par semaine
    rep_window: int = 8           # semaines d'avis utilisées pour le score de réputation
    min_margin: float = 0.10      # marge minimale imposée au-dessus du coût
    list_bounds: tuple = (0.80, 1.25)   # bornes du prix autour du prix catalogue
    step_sensitive: float = 0.05  # variation hebdo max (produits sensibles au prix)
    step_insensitive: float = 0.10  # variation hebdo max (produits peu sensibles)
    comp_cap_base: float = 1.05   # prix max = prix concurrent x (comp_cap_base + 0.3*(0.5 - réputation concurrent))
    lstm_epochs: int = 60
    shrink: float = 0.0           # poids du prior de segment sur l'élasticité (0 = régression seule)
    output_dir: str = "outputs"


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def has_transformers() -> bool:
    return has_torch() and importlib.util.find_spec("transformers") is not None


def resolve_backend(requested: str, kind: str) -> str:
    """Retourne 'torch' ou 'light' selon la demande ('auto' | 'torch' | 'light').

    kind = 'lstm' (nécessite torch) ou 'hf' (nécessite torch + transformers).
    'light' = modèles de repli scikit-learn (MLP, TF-IDF + régression logistique).
    """
    available = has_torch() if kind == "lstm" else has_transformers()
    if requested == "light":
        return "light"
    if requested == "torch" and not available:
        raise RuntimeError(
            "Backend 'torch' demandé mais torch/transformers ne sont pas installés "
            "(pip install -r requirements.txt)."
        )
    return "torch" if available else "light"
