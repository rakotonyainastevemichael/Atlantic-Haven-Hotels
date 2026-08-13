"""Feature engineering pour la prédiction d'annulation.

Deux niveaux sont distingués :

* `engineer()` — transformations **ligne à ligne**, sans aucun paramètre
  appris : aucune fuite possible, applicables indifféremment au train ou au
  test.
* `fit_reference()` / `engineer()` avec `reference` — variables nécessitant
  des statistiques agrégées (prix médian par hôtel, prix médian par
  région × mois, fréquences). Ces statistiques sont **apprises uniquement sur
  le pli d'entraînement** puis appliquées telles quelles à la validation et au
  test. Aucune n'utilise la cible : le risque de fuite est structurellement
  écarté.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ORDRE_ACOMPTE = {"aucun": 0, "partiel": 1, "total": 2}


# --------------------------------------------------------------------------- #
# Statistiques de référence (apprises sur le train uniquement)
# --------------------------------------------------------------------------- #
def fit_reference(train: pd.DataFrame) -> dict:
    """Apprend les statistiques agrégées à partir du seul jeu d'entraînement."""
    base = engineer(train, reference=None)
    return {
        "prix_hotel": base.groupby("hotel_id")["prix_moyen_nuit_eur"]
        .median()
        .to_dict(),
        "prix_region_mois": base.groupby(["region_hotel", "mois_arrivee"])[
            "prix_moyen_nuit_eur"
        ]
        .median()
        .to_dict(),
        "prix_global": float(base["prix_moyen_nuit_eur"].median()),
        "categories": {
            col: set(base[col].astype(str).unique())
            for col in [
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
        },
    }


# --------------------------------------------------------------------------- #
# Construction des variables
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame, reference: dict | None = None) -> pd.DataFrame:
    """Ajoute les variables dérivées au DataFrame nettoyé."""
    d = df.copy()

    # ---------------- Historique client ----------------
    # Un client ayant déjà annulé par le passé n'est comparable qu'à volume
    # de réservations équivalent : on raisonne en taux, pas en compte brut.
    reservations = d["reservations_passees"].where(d["reservations_passees"] > 0)
    d["taux_annulation_passe"] = (d["annulations_passees"] / reservations).fillna(0.0)
    d["sans_historique"] = (d["reservations_passees"] == 0).astype("int8")

    # ---------------- Composition du séjour ----------------
    d["personnes"] = d["adultes"] + d["enfants"]
    d["personnes_par_chambre"] = d["personnes"] / d["chambres"]
    d["avec_enfants"] = (d["enfants"] > 0).astype("int8")

    # ---------------- Prix et valeur ----------------
    d["prix_par_personne"] = d["montant_total_eur"] / d["personnes"].clip(lower=1)
    d["montant_par_nuit"] = d["montant_total_eur"] / d["nuits"]

    # ---------------- Engagement commercial ----------------
    # `score_engagement` synthétise en une échelle unique le niveau
    # d'irréversibilité de la réservation : acompte total non remboursable
    # (= 2) vs aucun acompte et tarif remboursable (= -1).
    d["acompte_ord"] = d["type_acompte"].map(ORDRE_ACOMPTE).astype(float)
    d["remboursable"] = (d["tarif_remboursable"] == "oui").astype("int8")
    d["score_engagement"] = d["acompte_ord"] - d["remboursable"]
    d["flexibilite_totale"] = (
        (d["acompte_ord"] == 0) & (d["remboursable"] == 1)
    ).astype("int8")

    # ---------------- Dimension temporelle ----------------
    d["mois_arrivee"] = d["date_arrivee"].dt.month
    d["mois_reservation"] = d["date_reservation"].dt.month
    # Encodage cyclique : décembre et janvier doivent être voisins.
    d["sin_mois"] = np.sin(2 * np.pi * d["mois_arrivee"] / 12)
    d["cos_mois"] = np.cos(2 * np.pi * d["mois_arrivee"] / 12)
    # Le délai est très asymétrique (médiane 29 j, max 584 j) : passage au log.
    d["log_delai"] = np.log1p(d["delai_reservation_jours"])
    d["delai_par_nuit"] = d["delai_reservation_jours"] / d["nuits"]

    # ---------------- Comportement ----------------
    d["a_modifie"] = (d["modifications_reservation"] > 0).astype("int8")

    # ---------------- Interactions explicites ----------------
    # L'EDA montre que l'effet du délai est **multiplicatif** : il pèse peu sur
    # un acompte total (10 % -> 14 % d'annulation) et beaucoup sans acompte
    # (29 % -> 49 %). Une régression logistique purement additive ne peut pas
    # capter cela : on l'aide avec des produits explicites.
    d["ix_delai_acompte"] = d["log_delai"] * d["acompte_ord"]
    d["ix_delai_remb"] = d["log_delai"] * d["remboursable"]
    d["ix_delai_direct"] = d["log_delai"] * d["reservation_directe"]
    d["ix_remb_acompte"] = d["remboursable"] * d["acompte_ord"]
    d["ix_delai_modif"] = d["log_delai"] * d["a_modifie"]
    d["ix_delai_dem"] = d["log_delai"] * d["demandes_speciales"]

    # ---------------- Variables nécessitant des références ----------------
    if reference is not None:
        prix_ref_hotel = d["hotel_id"].map(reference["prix_hotel"]).fillna(
            reference["prix_global"]
        )
        d["prix_relatif_hotel"] = d["prix_moyen_nuit_eur"] / prix_ref_hotel

        cle_saison = pd.Series(
            list(zip(d["region_hotel"], d["mois_arrivee"])), index=d.index
        )
        prix_ref_saison = cle_saison.map(reference["prix_region_mois"]).fillna(
            reference["prix_global"]
        )
        d["prix_relatif_saison"] = d["prix_moyen_nuit_eur"] / prix_ref_saison

        # Catégories jamais vues à l'entraînement -> modalité de repli.
        for col, connues in reference["categories"].items():
            d[col] = np.where(d[col].astype(str).isin(connues), d[col].astype(str), "AUTRE")

    return d


def build_matrix(
    df: pd.DataFrame, reference: dict, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Retourne la matrice de features prête pour un estimateur scikit-learn."""
    enriched = engineer(df, reference=reference)
    return enriched[numeric + categorical]
