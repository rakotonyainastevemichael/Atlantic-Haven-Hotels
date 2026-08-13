"""Configuration centrale du projet Atlantic Haven Hotels.

Toutes les constantes (chemins, graines, listes de colonnes, hyperparamètres)
sont regroupées ici afin que le pipeline reste reproductible et qu'aucune
valeur magique ne soit dispersée dans le code.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ressources"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "reservations_train.csv"
TEST_PATH = DATA_DIR / "reservations_test.csv"
SUBMISSION_PATH = ROOT / "submission.csv"

for _d in (FIG_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Colonnes
# --------------------------------------------------------------------------- #
TARGET = "reservation_annulee"
ID_COL = "reservation_id"
DATE_COLS = ["date_reservation", "date_arrivee"]

# Variables catégorielles retenues pour la modélisation.
# `ville` et `type_destination` sont volontairement exclues : elles sont en
# correspondance 1-à-1 avec `region_hotel` (vérifié dans l'EDA) et
# n'apporteraient qu'une redondance parfaite.
CATEGORICAL = [
    "type_acompte",
    "tarif_remboursable",
    "canal_reservation",
    "segment_client",
    "marche_origine",
    "moyen_transport",
    "formule_repas",
    "client_type",
    "region_hotel",
    "categorie_hotel",
]

# Variables numériques brutes du fichier source (utilisées par la baseline).
RAW_NUMERIC = [
    "delai_reservation_jours",
    "nuits",
    "adultes",
    "enfants",
    "chambres",
    "prix_moyen_nuit_eur",
    "remise_pct",
    "montant_total_eur",
    "reservations_passees",
    "annulations_passees",
    "demandes_speciales",
    "modifications_reservation",
    "jours_liste_attente",
    "evenement_majeur",
    "haute_saison_regionale",
    "arrivee_weekend",
]

# Variables numériques du modèle enrichi (brutes + feature engineering).
# Ce jeu correspond à la configuration « G » de l'étude d'ablation
# (`ablation.py`) : c'est la meilleure des huit configurations testées sur le
# protocole de validation temporelle. Les interactions explicites
# (configuration « H », listées dans `INTERACTION_FEATURES`) ont été testées
# puis **écartées** : elles améliorent marginalement l'AUC mais dégradent le
# F1 hors-échantillon.
MODEL_NUMERIC = [
    # --- variables brutes conservées
    "delai_reservation_jours",
    "nuits",
    "adultes",
    "enfants",
    "chambres",
    "prix_moyen_nuit_eur",
    "remise_pct",
    "montant_total_eur",
    "reservations_passees",
    "annulations_passees",
    "demandes_speciales",
    "modifications_reservation",
    "jours_liste_attente",
    "evenement_majeur",
    "haute_saison_regionale",
    "arrivee_weekend",
    # --- indicateurs de valeurs manquantes
    "prix_impute",
    "enfants_manquant",
    "demandes_manquant",
    "reservation_directe",
    # --- historique client
    "taux_annulation_passe",
    "sans_historique",
    # --- composition du séjour
    "personnes",
    "personnes_par_chambre",
    "avec_enfants",
    "prix_par_personne",
    "montant_par_nuit",
    # --- dimension temporelle
    "log_delai",
    "delai_par_nuit",
    "sin_mois",
    "cos_mois",
    # --- prix relatifs (références apprises sur le train uniquement)
    "prix_relatif_hotel",
    "prix_relatif_saison",
    # --- engagement commercial
    "score_engagement",
    "flexibilite_totale",
    "a_modifie",
]

# Testées puis écartées (voir ablation.py, configuration H).
INTERACTION_FEATURES = [
    "ix_delai_acompte",
    "ix_delai_remb",
    "ix_delai_direct",
    "ix_remb_acompte",
    "ix_delai_modif",
    "ix_delai_dem",
]

# --------------------------------------------------------------------------- #
# Protocole de validation temporelle
# --------------------------------------------------------------------------- #
N_FOLDS = 4          # nombre de plis à fenêtre étendue
VAL_FRACTION = 0.15  # part du train servant de validation à chaque pli

# --------------------------------------------------------------------------- #
# Hyperparamètres des modèles
# --------------------------------------------------------------------------- #
# Aucun `class_weight="balanced"` : le déséquilibre est traité par le seuil de
# décision. À F1 équivalent, cela préserve la calibration des probabilités
# (Brier 0,178 contre 0,225 avec rééquilibrage) — indispensable puisque le
# livrable exige une probabilité d'annulation exploitable telle quelle.
LOGREG_PARAMS = dict(
    C=0.05,
    max_iter=5000,
    solver="lbfgs",
    random_state=SEED,
)

RF_PARAMS = dict(
    n_estimators=500,
    min_samples_leaf=30,
    max_features=0.3,
    random_state=SEED,
    n_jobs=-1,
)

HGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=8,
    min_samples_leaf=60,
    l2_regularization=5.0,
    random_state=SEED,
)

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=80,
    colsample_bytree=0.6,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=10.0,
    random_state=SEED,
    verbose=-1,
)

# Pondération de l'ensemble final (moyenne pondérée des probabilités).
ENSEMBLE_WEIGHTS = {"logreg": 0.6, "gbm": 0.4}

# Grille de recherche du seuil de décision.
THRESHOLD_GRID = (0.05, 0.95, 181)
