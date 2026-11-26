"""Tableau de bord de démonstration (Streamlit) : `streamlit run app_streamlit.py`."""
import streamlit as st

from ecom_agents.config import Config
from ecom_agents.data import generate_world
from ecom_agents.orchestrator import Orchestrator


@st.cache_resource(max_entries=1, show_spinner="Génération des données...")
def load_world():
    cfg = Config()
    return generate_world(cfg)


@st.cache_resource(max_entries=2, show_spinner="Entraînement des agents...")
def load_orchestrator(backend: str):
    cfg = Config()
    world = load_world()
    orch = Orchestrator(cfg, world, backend).fit()
    state = orch.initial_state()
    for wk in range(orch.t0, orch.t0 + 3):
        p, _ = orch.decide(wk, state)
        state.units[:, wk], state.price[:, wk] = world.demand(wk, p), p
    return orch, state, orch.t0 + 3


_fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", lambda fn: fn))


@_fragment
def whatif_panel(orch, state, w: int):
    i = st.selectbox(
        "Produit",
        range(orch.cfg.n_products),
        format_func=lambda k: f"{k} - {orch.world.products.loc[k, 'category']}",
        key="product_id",
    )
    comp_chg = st.slider("Variation du prix concurrent (%)", -30, 30, 0, key="comp_chg")
    rep_chg = st.slider("Variation de la réputation concurrent", -0.4, 0.4, 0.0, 0.05, key="rep_chg")
    _, dec = orch.decide(w, state.copy(), {"comp_mult": {i: 1 + comp_chg / 100}, "rep_delta": {i: rep_chg}})
    pr = orch.messages_for(i, dec)[-1]["payload"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Prix recommandé", f"{pr['prix_recommande']:.2f}", f"{pr['prix_recommande'] - pr['prix_precedent']:+.2f}")
    c2.metric("Prix concurrent", f"{dec['pc_prev'][i]:.2f}")
    c3.metric("Réputation concurrent", f"{dec['rep'][i]:.2f}")
    st.caption(f"Contrainte active : {pr['contrainte_active']} | élasticité estimée : {pr['elasticite']:.2f}")
    st.subheader("Messages échangés entre agents")
    st.json(orch.messages_for(i, dec))


st.title("Tarification dynamique multi-agents")
if hasattr(st, "segmented_control"):
    backend = st.segmented_control("Backend des modèles", ["light", "torch"], default="light", key="backend") or "light"
else:
    backend = st.radio("Backend des modèles", ["light", "torch"], index=0, horizontal=True, key="backend")

if backend == "torch":
    st.warning("Le mode torch exécute DistilBERT (~37 000 avis) et le LSTM : le premier entraînement prend plusieurs minutes sur CPU.")

content_slot = st.container()
with content_slot:
    orch, state, w = load_orchestrator(backend)
    whatif_panel(orch, state, w)
