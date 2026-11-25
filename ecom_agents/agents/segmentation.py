"""Agent Segmentation : arbres de décision (règles lisibles) + Random Forest (performance)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from .base import Agent


class SegmentationAgent(Agent):
    """Classe chaque produit en 'sensible au prix' (1) ou 'peu sensible' (0).

    Cible = élasticité estimée par régression sous la médiane du catalogue. L'intérêt du classifieur
    est le *démarrage à froid* (nouveau produit sans historique) et la règle interprétable."""

    name = "segmentation"

    def _features(self, world, rep_hat, t0) -> pd.DataFrame:
        lq = np.log(world.units_hist[:, :t0])
        lp = np.log(world.price_hist[:, :t0])
        df = pd.DataFrame({
            "log_prix": np.log(world.p_ref),
            "taux_marge": 1 - world.cost / world.p_ref,
            "log_volume": lq.mean(1),
            "volatilite_ventes": lq.std(1),
            "prix_vs_concurrent": (world.price_hist[:, :t0] / world.comp_price[:, :t0]).mean(1),
            "reputation_moy": rep_hat[:, :t0].mean(1),
            "freq_promos": (np.abs(np.diff(lp, axis=1)) > 0.03).mean(1),
        })
        dummies = pd.get_dummies(world.products["category"], prefix="cat").astype(int)
        return pd.concat([df, dummies], axis=1)

    def fit(self, world, rep_hat, eps_hat, t0, seed=0):
        X = self._features(world, rep_hat, t0)
        self.cols = list(X.columns)
        y = (eps_hat < np.median(eps_hat)).astype(int)      # 1 = sensible au prix
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=seed)
        rf = RandomForestClassifier(300, max_depth=6, min_samples_leaf=2, random_state=seed)
        self.metrics = {"cv_accuracy_decision_tree": float(cross_val_score(dt, X, y, cv=cv).mean()),
                        "cv_accuracy_random_forest": float(cross_val_score(rf, X, y, cv=cv).mean())}
        # prédictions hors-échantillon : évite de réutiliser l'étiquette apprise sur le même produit
        self.proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
        self.sensitive = self.proba > 0.5
        self.dt, self.rf = dt.fit(X, y), rf.fit(X, y)
        self.rules = export_text(self.dt, feature_names=self.cols)
        self.importances = dict(sorted(zip(self.cols, map(float, self.rf.feature_importances_)), key=lambda kv: -kv[1]))
        return self

    def run_batch(self) -> dict:
        return {"sensitive": self.sensitive, "proba": self.proba}
