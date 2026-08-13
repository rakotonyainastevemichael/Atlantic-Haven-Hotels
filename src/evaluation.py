"""Métriques, choix du seuil de décision et analyse d'erreurs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C


def threshold_grid() -> np.ndarray:
    lo, hi, n = C.THRESHOLD_GRID
    return np.linspace(lo, hi, n)


def best_threshold(y_true, proba) -> tuple[float, float]:
    """Seuil maximisant le F1 sur la classe positive, et le F1 associé."""
    grid = threshold_grid()
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def metrics(y_true, proba, threshold: float) -> dict:
    """Panorama de métriques à un seuil donné."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "seuil": round(float(threshold), 3),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "rappel": recall_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "vp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "vn": int(tn),
        "taux_alerte": float(pred.mean()),
    }


def pooled_threshold(oof: pd.DataFrame) -> tuple[float, float]:
    """Seuil unique choisi sur l'ensemble des prédictions hors-échantillon.

    Optimiser le seuil sur chaque pli séparément puis moyenner surestime la
    performance ; on agrège au contraire les prédictions out-of-fold des
    quatre plis temporels et on cherche **un seul** seuil, celui qui sera
    effectivement appliqué au jeu de test.
    """
    return best_threshold(oof["y"].values, oof["proba"].values)


def summarize_folds(records: list[dict]) -> pd.DataFrame:
    """Moyenne et écart-type des métriques par modèle, sur les plis temporels."""
    df = pd.DataFrame(records)
    agg = (
        df.groupby("modele", sort=False)[["f1", "precision", "rappel", "roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# Analyse d'erreurs
# --------------------------------------------------------------------------- #
def error_frame(df: pd.DataFrame, y_true, proba, threshold: float) -> pd.DataFrame:
    """Annote chaque réservation avec son type d'erreur."""
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["y_reel"] = np.asarray(y_true)
    out["proba"] = proba
    out["y_predit"] = pred
    conditions = [
        (out.y_reel == 1) & (out.y_predit == 1),
        (out.y_reel == 0) & (out.y_predit == 1),
        (out.y_reel == 1) & (out.y_predit == 0),
    ]
    out["type_erreur"] = np.select(
        conditions, ["vrai positif", "faux positif", "faux négatif"], "vrai négatif"
    )
    return out


def performance_par_groupe(
    errors: pd.DataFrame, colonne: str, min_effectif: int = 60
) -> pd.DataFrame:
    """F1 par sous-groupe, avec effectif, pour discuter l'équité du modèle."""
    rows = []
    for valeur, g in errors.groupby(colonne, observed=True):
        if len(g) < min_effectif:
            continue
        rows.append(
            {
                colonne: valeur,
                "effectif": len(g),
                "taux_annulation_reel": round(float(g.y_reel.mean()), 3),
                "f1": round(f1_score(g.y_reel, g.y_predit, zero_division=0), 3),
                "precision": round(
                    precision_score(g.y_reel, g.y_predit, zero_division=0), 3
                ),
                "rappel": round(recall_score(g.y_reel, g.y_predit, zero_division=0), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def cout_metier(
    y_true, proba, cout_fn: float = 1.0, cout_fp: float = 0.25
) -> pd.DataFrame:
    """Courbe de coût espéré selon le seuil.

    `cout_fn` : chambre restée vide faute d'anticipation.
    `cout_fp` : geste commercial inutile envoyé à un client fidèle.
    Le rapport 4:1 retenu par défaut est une hypothèse de travail explicite,
    à recalibrer avec le revenue manager.
    """
    rows = []
    for t in threshold_grid():
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        rows.append(
            {
                "seuil": round(float(t), 3),
                "fp": int(fp),
                "fn": int(fn),
                "cout": cout_fp * fp + cout_fn * fn,
                "f1": f1_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)


def coefficients_logreg(pipeline, numeric, categorical) -> pd.DataFrame:
    """Coefficients standardisés d'une régression logistique en pipeline."""
    prep = pipeline.named_steps["prep"]
    noms = list(numeric) + list(
        prep.named_transformers_["cat"].get_feature_names_out(categorical)
    )
    coefs = pipeline.named_steps["clf"].coef_[0]
    return (
        pd.DataFrame({"variable": noms, "coefficient": coefs})
        .assign(odds_ratio=lambda d: np.exp(d.coefficient))
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )
