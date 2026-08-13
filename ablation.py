#!/usr/bin/env python3
"""Étude d'ablation : quel groupe de variables créées apporte réellement un gain ?

Chaque configuration est évaluée avec **exactement le même protocole** que le
pipeline principal (4 plis temporels à fenêtre étendue, prédictions
hors-échantillon agrégées, seuil optimisé sur ces prédictions), afin que les
écarts observés soient comparables.

Exécution :
    python ablation.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config as C
from src import data as D
from src import evaluation as E
from src import features as F
from src import models as M

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Groupes de variables, ajoutés de façon cumulative
# --------------------------------------------------------------------------- #
G_BRUT = [
    "delai_reservation_jours", "nuits", "adultes", "enfants", "chambres",
    "prix_moyen_nuit_eur", "remise_pct", "montant_total_eur", "reservations_passees",
    "annulations_passees", "demandes_speciales", "modifications_reservation",
    "jours_liste_attente", "evenement_majeur", "haute_saison_regionale",
    "arrivee_weekend",
]
G_MANQUANTS = ["prix_impute", "enfants_manquant", "demandes_manquant", "reservation_directe"]
G_HISTORIQUE = ["taux_annulation_passe", "sans_historique"]
G_SEJOUR = ["personnes", "personnes_par_chambre", "avec_enfants", "prix_par_personne",
            "montant_par_nuit"]
G_TEMPS = ["log_delai", "delai_par_nuit", "sin_mois", "cos_mois"]
G_PRIX_REL = ["prix_relatif_hotel", "prix_relatif_saison"]
G_ENGAGEMENT = ["score_engagement", "flexibilite_totale", "a_modifie"]
G_INTERACTIONS = ["ix_delai_acompte", "ix_delai_remb", "ix_delai_direct",
                  "ix_remb_acompte", "ix_delai_modif", "ix_delai_dem"]

CONFIGS = {
    "A. Variables brutes (baseline)": G_BRUT,
    "B. + indicateurs de manque": G_BRUT + G_MANQUANTS,
    "C. + historique client": G_BRUT + G_MANQUANTS + G_HISTORIQUE,
    "D. + composition du séjour": G_BRUT + G_MANQUANTS + G_HISTORIQUE + G_SEJOUR,
    "E. + transformations temporelles": G_BRUT + G_MANQUANTS + G_HISTORIQUE + G_SEJOUR + G_TEMPS,
    "F. + prix relatifs": G_BRUT + G_MANQUANTS + G_HISTORIQUE + G_SEJOUR + G_TEMPS + G_PRIX_REL,
    "G. + score d'engagement": G_BRUT + G_MANQUANTS + G_HISTORIQUE + G_SEJOUR + G_TEMPS
    + G_PRIX_REL + G_ENGAGEMENT,
    "H. + interactions explicites": G_BRUT + G_MANQUANTS + G_HISTORIQUE + G_SEJOUR + G_TEMPS
    + G_PRIX_REL + G_ENGAGEMENT + G_INTERACTIONS,
}


def evaluate_config(train: pd.DataFrame, folds, numeric: list[str]) -> dict:
    """Prédictions hors-échantillon d'une régression logistique sur ces variables."""
    parts = []
    for itr, iva in folds:
        reference = F.fit_reference(train.iloc[itr])
        TR = F.engineer(train.iloc[itr], reference)
        VA = F.engineer(train.iloc[iva], reference)
        model = Pipeline(
            [
                ("prep", M.make_linear_preprocessor(numeric, C.CATEGORICAL)),
                ("clf", LogisticRegression(**C.LOGREG_PARAMS)),
            ]
        )
        cols = numeric + C.CATEGORICAL
        model.fit(TR[cols], TR[C.TARGET].values)
        parts.append(
            pd.DataFrame(
                {"y": VA[C.TARGET].values, "proba": model.predict_proba(VA[cols])[:, 1]}
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    seuil, _ = E.best_threshold(oof.y.values, oof.proba.values)
    return E.metrics(oof.y.values, oof.proba.values, seuil)


def main() -> None:
    train, _ = D.load_train_test()
    folds = D.temporal_folds(train)

    rows = []
    reference_f1 = None
    for nom, numeric in CONFIGS.items():
        m = evaluate_config(train, folds, numeric)
        if reference_f1 is None:
            reference_f1 = m["f1"]
        rows.append(
            {
                "configuration": nom,
                "n_variables": len(numeric),
                "f1": round(m["f1"], 4),
                "gain_vs_baseline": round(m["f1"] - reference_f1, 4),
                "precision": round(m["precision"], 4),
                "rappel": round(m["rappel"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "seuil": m["seuil"],
            }
        )
        print(f"{nom:36s} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    table = pd.DataFrame(rows)
    table["gain_marginal"] = table["f1"].diff().round(4)
    table.to_csv(C.OUT_DIR / "ablation_feature_engineering.csv", index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nTable enregistrée : {C.OUT_DIR / 'ablation_feature_engineering.csv'}")


if __name__ == "__main__":
    main()
