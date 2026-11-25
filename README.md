# Projet 8 — Tarification dynamique & concurrence en e-commerce (architecture multi-agents)

Système de 5 agents coordonnés par un orchestrateur, chaque agent intégrant un modèle vu en cours.

| Agent | Rôle | Modèles |
|---|---|---|
| Demande | prévoit les ventes de la semaine suivante | **LSTM** (repli : MLP scikit-learn) + baseline Ridge |
| Réputation | score de réputation du concurrent à partir de ses avis | **Transformer** (DistilBERT, repli : TF-IDF+LogReg) + **Naive Bayes** (baseline) |
| Vision | attributs visuels des produits (optionnel, nécessite des images) | **CNN** |
| Segmentation | produit sensible / peu sensible au prix | **Arbre de décision** (règles) + **Random Forest** |
| Pricing | élasticité-prix puis prix optimal sous contraintes | **Régression linéaire** (log-log) + optimisation de marge |

Orchestrateur : exécute Demande / Réputation / Vision en parallèle, puis Segmentation, puis Pricing,
applique les garde-fous (plancher de marge, variation hebdo max, plafond lié au concurrent) et journalise
chaque décision (messages JSON) dans `outputs/decisions.jsonl`.

## Installation et exécution
```bash
pip install -r requirements.txt
python run_all.py                 # pipeline complet -> outputs/ (results.json, figures/, decisions.jsonl)
python run_all.py --cap 1.15      # variante : plafond concurrent à +15 %
python run_seeds.py --seeds 5     # robustesse (5 graines) + ablations
python demo_whatif.py             # démonstration en ligne de commande (what-if)
streamlit run app_streamlit.py    # démonstration interactive (non testée dans l'environnement de développement)
```
`--backend light` force les modèles de repli scikit-learn ; `--backend torch` exige PyTorch + transformers.

## État de validation (important)
* Le pipeline complet a été **exécuté et mesuré avec le backend `light`** (MLP à la place du LSTM, TF-IDF+LogReg à la place
  du Transformer). Les chemins **PyTorch/Hugging Face (LSTM, DistilBERT, CNN) et Streamlit sont écrits mais n'ont pas pu
  être exécutés** (pas de PyTorch ni d'accès réseau dans l'environnement de développement) : à lancer et à vérifier chez vous.
* Les jeux Kaggle n'étaient pas téléchargeables : les données sont **synthétiques** (`ecom_agents/data.py`), avec paramètres
  vrais connus pour permettre un backtest. `world_from_tables` permet de brancher les vraies données (format en docstring) ;
  dans ce cas le backtest contrefactuel n'est pas possible (seule l'évaluation des prévisions/classifieurs l'est).

## Structure
```
ecom_agents/  config.py  data.py  nn_models.py  orchestrator.py  evaluation.py
              agents/ base.py demand.py reputation.py vision.py segmentation.py pricing.py
run_all.py  run_seeds.py  demo_whatif.py  app_streamlit.py
outputs/ (plafond +5 %, graine 42)   outputs_cap115/ (plafond +15 %)
```
