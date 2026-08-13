"""Chargement et nettoyage des données Atlantic Haven Hotels.

Règle centrale : **aucun paramètre appris ici ne dépend de la cible ni du jeu
de test**. Les imputations réalisées dans `clean()` sont soit déterministes
(reconstruction exacte du prix à partir du montant total), soit des constantes
neutres accompagnées d'un indicateur de manque. Les statistiques réellement
apprises (médianes de prix par hôtel, fréquences) sont isolées dans
`features.fit_reference()` et ajustées uniquement sur le pli d'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw(path) -> pd.DataFrame:
    """Charge un CSV du hackathon en typant correctement les dates."""
    return pd.read_csv(path, parse_dates=C.DATE_COLS)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les jeux d'entraînement et de test nettoyés."""
    train = clean(load_raw(C.TRAIN_PATH)).reset_index(drop=True)
    test = clean(load_raw(C.TEST_PATH)).reset_index(drop=True)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un jeu de réservations.

    Traitements appliqués (tous justifiés dans le README) :

    1. `prix_moyen_nuit_eur` manquant est **reconstruit exactement** via
       l'identité comptable vérifiée sur 7 807 lignes du train :
       ``montant_total = prix * nuits * chambres * (1 - remise/100)``.
       Un indicateur `prix_impute` conserve la trace de l'opération.
    2. `enfants` et `demandes_speciales` manquants sont remplis par 0
       (modalité majoritaire et sémantiquement neutre) + indicateur.
    3. `marche_origine` manquant devient une catégorie explicite `Inconnu`
       plutôt que d'être supprimé : le manque est lui-même informatif.
    4. `agent_id` vide signifie « réservation directe » d'après le
       dictionnaire de données : on crée `reservation_directe` et on remplit
       par la modalité `DIRECT`.
    """
    d = df.copy()

    # 1. Prix reconstruit par identité comptable (déterministe, sans fuite).
    denominateur = d["nuits"] * d["chambres"] * (1 - d["remise_pct"] / 100)
    prix_reconstruit = d["montant_total_eur"] / denominateur
    d["prix_impute"] = d["prix_moyen_nuit_eur"].isna().astype("int8")
    d["prix_moyen_nuit_eur"] = d["prix_moyen_nuit_eur"].fillna(prix_reconstruit)

    # 2. Manques neutres + indicateurs.
    d["enfants_manquant"] = d["enfants"].isna().astype("int8")
    d["enfants"] = d["enfants"].fillna(0)
    d["demandes_manquant"] = d["demandes_speciales"].isna().astype("int8")
    d["demandes_speciales"] = d["demandes_speciales"].fillna(0)

    # 3. Modalité explicite pour le marché d'origine.
    d["marche_origine"] = d["marche_origine"].fillna("Inconnu")

    # 4. Absence d'agent = réservation directe.
    d["reservation_directe"] = d["agent_id"].isna().astype("int8")
    d["agent_id"] = d["agent_id"].fillna("DIRECT")

    # Sécurité : la catégorie d'hôtel est un entier mais se comporte comme un
    # facteur ordinal à trois modalités -> traitée comme catégorielle.
    d["categorie_hotel"] = d["categorie_hotel"].astype(str)

    return d


def temporal_folds(
    df: pd.DataFrame,
    n_folds: int = C.N_FOLDS,
    val_fraction: float = C.VAL_FRACTION,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construit des plis de validation à **fenêtre étendue** (expanding window).

    Le jeu de test représentant des réservations plus récentes que le train,
    une validation croisée aléatoire mélangerait passé et futur et produirait
    une estimation optimiste. Chaque pli entraîne donc sur tout le passé
    disponible et valide sur le bloc temporel qui suit immédiatement.

    Les données étant déjà triées par `date_reservation` (vérifié dans l'EDA),
    un découpage par position équivaut à un découpage par date.
    """
    n = len(df)
    size = int(n * val_fraction)
    folds = []
    for i in range(n_folds):
        end = int(n * (1 - val_fraction * (n_folds - 1 - i)))
        start = end - size
        folds.append((np.arange(0, start), np.arange(start, end)))
    return folds


def fold_dates(df: pd.DataFrame, folds) -> pd.DataFrame:
    """Résume les bornes temporelles de chaque pli (pour le rapport)."""
    rows = []
    for k, (itr, iva) in enumerate(folds, start=1):
        rows.append(
            {
                "pli": k,
                "train_debut": df["date_reservation"].iloc[itr[0]].date(),
                "train_fin": df["date_reservation"].iloc[itr[-1]].date(),
                "n_train": len(itr),
                "val_debut": df["date_reservation"].iloc[iva[0]].date(),
                "val_fin": df["date_reservation"].iloc[iva[-1]].date(),
                "n_val": len(iva),
                "taux_annulation_val": round(
                    float(df[C.TARGET].iloc[iva].mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)
