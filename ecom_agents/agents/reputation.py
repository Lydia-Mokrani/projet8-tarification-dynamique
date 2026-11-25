"""Agent Réputation : Transformer (analyse d'avis) + Naive Bayes (baseline)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from ..config import Config, resolve_backend
from .base import Agent, Message

HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class NaiveBayesSentiment:
    """Baseline rapide : sacs de mots + Naive Bayes multinomial."""

    def fit(self, texts, y):
        self.vec = CountVectorizer(min_df=2)
        self.clf = MultinomialNB()
        self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


class TransformerSentiment:
    """Backend 'torch' : DistilBERT pré-entraîné (SST-2) via Hugging Face.
    Backend 'light'  : TF-IDF + régression logistique (repli quand transformers est absent)."""

    def __init__(self, backend: str = "auto"):
        self.backend = resolve_backend(backend, "hf")

    def fit(self, texts, y):
        if self.backend == "torch":
            from transformers import pipeline
            self.pipe = pipeline("sentiment-analysis", model=HF_MODEL, truncation=True)
        else:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            self.clf = LogisticRegression(max_iter=1000)
            self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if self.backend == "torch":
            out = self.pipe(list(texts), batch_size=64)
            return np.array([o["score"] if o["label"] == "POSITIVE" else 1 - o["score"] for o in out])
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


class ReputationAgent(Agent):
    name = "reputation"

    def __init__(self, cfg: Config, backend: str = "auto"):
        self.cfg = cfg
        self.nb = NaiveBayesSentiment()
        self.main = TransformerSentiment(backend)
        self.metrics: dict = {}

    def fit(self, reviews, t0: int):
        """Entraîne et compare les deux classifieurs sur les avis de la période d'apprentissage."""
        tr = reviews[(reviews.week < t0) & (reviews.rating != 3)]
        texts, y = tr.text.to_numpy(), (tr.rating.to_numpy() >= 4).astype(int)
        perm = np.random.default_rng(self.cfg.seed).permutation(len(texts))
        cut = int(0.8 * len(texts))
        a, b = perm[:cut], perm[cut:]
        self.nb.fit(texts[a], y[a])
        self.main.fit(texts[a], y[a])
        for name, model in (("naive_bayes", self.nb), (f"transformer[{self.main.backend}]", self.main)):
            pred = (model.predict_proba(texts[b]) > 0.5).astype(int)
            self.metrics[name] = {"accuracy": float(accuracy_score(y[b], pred)), "f1": float(f1_score(y[b], pred))}
        return self

    def build_series(self, reviews, n: int, t: int) -> np.ndarray:
        """Score de réputation glissant (moyenne de P(avis positif) sur rep_window semaines), causal."""
        p = self.main.predict_proba(reviews.text.to_numpy())
        pid, wk = reviews.product_id.to_numpy(), reviews.week.to_numpy()
        s, c = np.zeros((n, t)), np.zeros((n, t))
        np.add.at(s, (pid, wk), p)
        np.add.at(c, (pid, wk), 1)
        win = self.cfg.rep_window
        cs, cc = np.cumsum(s, 1), np.cumsum(c, 1)
        sh_s, sh_c = np.zeros_like(cs), np.zeros_like(cc)
        sh_s[:, win:], sh_c[:, win:] = cs[:, :-win], cc[:, :-win]
        s_roll, c_roll = cs - sh_s, cc - sh_c
        return np.where(c_roll > 0, s_roll / np.maximum(c_roll, 1), 0.5)

    def run_batch(self, state, w: int) -> dict:
        rep = state.rep_hat[:, w - 1]
        trend = rep - state.rep_hat[:, max(w - 5, 0)]
        return {"rep": rep, "trend": trend}
