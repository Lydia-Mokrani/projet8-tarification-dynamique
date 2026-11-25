"""Tableau de bord de démonstration (Streamlit) : `streamlit run app_streamlit.py`."""
import numpy as np
import streamlit as st

from ecom_agents.config import Config
from ecom_agents.data import generate_world
from ecom_agents.orchestrator import Orchestrator


@st.cache_resource(show_spinner="Entraînement des agents...")
def load():
    cfg = Config()
    world = generate_world(cfg)
    orch = Orchestrator(cfg, world, "auto").fit()
    state = orch.initial_state()
    for w in range(orch.t0, orch.t0 + 3):
        p, _ = orch.decide(w, state)
        state.units[:, w], state.price[:, w] = world.demand(w, p), p
    return orch, state, orch.t0 + 3


orch, state, w = load()
st.title("Tarification dynamique multi-agents")
i = st.selectbox("Produit", range(orch.cfg.n_products), format_func=lambda k: f"{k} - {orch.world.products.loc[k, 'category']}")
comp_chg = st.slider("Variation du prix concurrent (%)", -30, 30, 0)
rep_chg = st.slider("Variation de la réputation concurrent", -0.4, 0.4, 0.0, 0.05)
prices, dec = orch.decide(w, state, {"comp_mult": {i: 1 + comp_chg / 100}, "rep_delta": {i: rep_chg}})
pr = orch.messages_for(i, dec)[-1]["payload"]
c1, c2, c3 = st.columns(3)
c1.metric("Prix recommandé", f"{pr['prix_recommande']:.2f}", f"{pr['prix_recommande'] - pr['prix_precedent']:+.2f}")
c2.metric("Prix concurrent", f"{dec['pc_prev'][i]:.2f}")
c3.metric("Réputation concurrent", f"{dec['rep'][i]:.2f}")
st.caption(f"Contrainte active : {pr['contrainte_active']} | élasticité estimée : {pr['elasticite']:.2f}")
st.subheader("Messages échangés entre agents")
st.json(orch.messages_for(i, dec))
