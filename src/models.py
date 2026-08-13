"""Définition des modèles comparés et de l'ensemble final.

Familles comparées :

1. **Régression logistique** (baseline imposée, puis version enrichie) ;
2. **Forêt aléatoire** — bagging, non paramétrique ;
3. **Gradient boosting histogramme** (scikit-learn) ;
4. **LightGBM** — boosting par feuilles, plus agressif.

Le modèle final est une **moyenne pondérée des probabilités** des trois
meilleurs modèles. Une moyenne de probabilités (et non de rangs) est retenue
car elle reste calculable ligne par ligne : la prédiction d'une réservation ne
dépend pas des autres réservations du lot, condition indispensable à une mise
en production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

try:  # LightGBM est optionnel : le pipeline reste exécutable sans lui.
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Prétraitements
# --------------------------------------------------------------------------- #
def make_linear_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Imputation médiane + standardisation + one-hot (modèles linéaires).

    `handle_unknown="ignore"` garantit qu'une modalité jamais vue à
    l'entraînement (par exemple le canal `assistant_vocal`, présent uniquement
    dans le test) produit un vecteur nul au lieu d'une erreur.
    """
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
        ]
    )


def make_tree_preprocessor(numeric, categorical) -> ColumnTransformer:
    """One-hot sans standardisation : inutile pour les modèles à base d'arbres."""
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


# --------------------------------------------------------------------------- #
# Constructeurs de modèles
# --------------------------------------------------------------------------- #
def build_baseline(numeric=None, categorical=None) -> Pipeline:
    """Baseline imposée : régression logistique sur les variables brutes."""
    numeric = numeric if numeric is not None else C.RAW_NUMERIC
    categorical = categorical if categorical is not None else C.CATEGORICAL
    return Pipeline(
        [
            ("prep", make_linear_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(**{**C.LOGREG_PARAMS, "C": 1.0})),
        ]
    )


def build_logreg(numeric=None, categorical=None) -> Pipeline:
    """Régression logistique enrichie (feature engineering + interactions)."""
    numeric = numeric if numeric is not None else C.MODEL_NUMERIC
    categorical = categorical if categorical is not None else C.CATEGORICAL
    return Pipeline(
        [
            ("prep", make_linear_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(**C.LOGREG_PARAMS)),
        ]
    )


def build_random_forest(numeric=None, categorical=None) -> Pipeline:
    numeric = numeric if numeric is not None else C.MODEL_NUMERIC
    categorical = categorical if categorical is not None else C.CATEGORICAL
    return Pipeline(
        [
            ("prep", make_tree_preprocessor(numeric, categorical)),
            ("clf", RandomForestClassifier(**C.RF_PARAMS)),
        ]
    )


def build_hgb(numeric=None, categorical=None) -> Pipeline:
    numeric = numeric if numeric is not None else C.MODEL_NUMERIC
    categorical = categorical if categorical is not None else C.CATEGORICAL
    return Pipeline(
        [
            ("prep", make_tree_preprocessor(numeric, categorical)),
            (
                "clf",
                HistGradientBoostingClassifier(**C.HGB_PARAMS),
            ),
        ]
    )


def build_lgbm(numeric=None, categorical=None) -> Pipeline | None:
    if not LIGHTGBM_AVAILABLE:
        return None
    numeric = numeric if numeric is not None else C.MODEL_NUMERIC
    categorical = categorical if categorical is not None else C.CATEGORICAL
    return Pipeline(
        [
            ("prep", make_tree_preprocessor(numeric, categorical)),
            ("clf", LGBMClassifier(**C.LGBM_PARAMS)),
        ]
    )


def build_all() -> dict[str, Pipeline]:
    """Dictionnaire ordonné des modèles à comparer sur le même protocole."""
    models = {
        "Régression logistique — baseline": build_baseline(),
        "Régression logistique + FE": build_logreg(),
        "Forêt aléatoire": build_random_forest(),
        "HistGradientBoosting": build_hgb(),
    }
    lgbm = build_lgbm()
    if lgbm is not None:
        models["LightGBM"] = lgbm
    return models


# --------------------------------------------------------------------------- #
# Ensemble final
# --------------------------------------------------------------------------- #
ENSEMBLE_MEMBERS = ("Régression logistique + FE", "HistGradientBoosting", "LightGBM")
ENSEMBLE_RAW_WEIGHTS = {
    "Régression logistique + FE": 0.50,
    "HistGradientBoosting": 0.25,
    "LightGBM": 0.25,
}


def ensemble_weights(available: list[str]) -> dict[str, float]:
    """Poids renormalisés sur les seuls modèles réellement disponibles."""
    w = {k: v for k, v in ENSEMBLE_RAW_WEIGHTS.items() if k in available}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def ensemble_proba(probas: dict[str, np.ndarray]) -> np.ndarray:
    """Moyenne pondérée des probabilités des membres de l'ensemble."""
    weights = ensemble_weights(list(probas))
    return sum(weights[k] * np.asarray(probas[k]) for k in weights)


class EnsembleAnnulation:
    """Modèle final réutilisable : entraînement, probabilité, décision.

    Encapsule les trois estimateurs, la table de référence du feature
    engineering et le seuil de décision, de sorte qu'une réservation isolée
    puisse être scorée en production avec exactement la même logique que
    pendant la validation.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.models_: dict[str, Pipeline] = {}
        self.reference_: dict | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, reference: dict) -> "EnsembleAnnulation":
        self.reference_ = reference
        builders = {
            "Régression logistique + FE": build_logreg,
            "HistGradientBoosting": build_hgb,
            "LightGBM": build_lgbm,
        }
        for name in ENSEMBLE_MEMBERS:
            model = builders[name]()
            if model is None:
                continue
            model.fit(X, y)
            self.models_[name] = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = {n: m.predict_proba(X)[:, 1] for n, m in self.models_.items()}
        return ensemble_proba(probas)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)
