#!/usr/bin/env python3
"""Pipeline complet Atlantic Haven Hotels — de la donnée brute à submission.csv.

Exécution :

    python run_pipeline.py

Produit :
    submission.csv            prédictions sur reservations_test.csv
    outputs/*.csv             tables de résultats reprises dans le README
    figures/*.png             graphiques de l'EDA et de l'évaluation
"""

from __future__ import annotations

import random
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from src import config as C
from src import data as D
from src import evaluation as E
from src import features as F
from src import models as M

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
random.seed(C.SEED)
np.random.seed(C.SEED)

PALETTE = {
    "laterite": "#C1440E",
    "ocre": "#E4A11B",
    "lagon": "#0E7C86",
    "encre": "#2B2118",
    "creme": "#F7EFE3",
}
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.facecolor": PALETTE["creme"],
        "figure.facecolor": "white",
        "axes.edgecolor": PALETTE["encre"],
        "axes.titleweight": "bold",
        "axes.titlecolor": PALETTE["encre"],
    }
)


def titre(texte: str) -> None:
    print("\n" + "=" * 78)
    print(texte)
    print("=" * 78)


def main() -> None:
    t0 = time.time()

    # ----------------------------------------------------------------- #
    # 1. Chargement et nettoyage
    # ----------------------------------------------------------------- #
    titre("1. CHARGEMENT ET NETTOYAGE")
    train, test = D.load_train_test()
    y = train[C.TARGET].values
    print(f"train : {train.shape[0]} lignes | test : {test.shape[0]} lignes")
    print(f"taux d'annulation global : {y.mean():.4f}")
    print(
        "période train : "
        f"{train.date_reservation.min().date()} -> {train.date_reservation.max().date()}"
    )
    print(
        "période test  : "
        f"{test.date_reservation.min().date()} -> {test.date_reservation.max().date()}"
    )

    # ----------------------------------------------------------------- #
    # 2. Quelques constats d'EDA sauvegardés
    # ----------------------------------------------------------------- #
    titre("2. EDA — CONSTATS SAUVEGARDÉS")
    eda = F.engineer(train)
    eda[C.TARGET] = y

    croisement = (
        eda.pivot_table(
            index="type_acompte", columns="tarif_remboursable", values=C.TARGET, aggfunc="mean"
        )
        .round(3)
    )
    croisement.to_csv(C.OUT_DIR / "eda_acompte_x_remboursable.csv")
    print("Taux d'annulation — acompte x tarif remboursable :")
    print(croisement)

    tranches = pd.cut(eda.delai_reservation_jours, [0, 7, 21, 45, 90, 600])
    delai_tab = (
        eda.groupby([tranches, "type_acompte"], observed=True)[C.TARGET]
        .mean()
        .unstack()
        .round(3)
    )
    delai_tab.to_csv(C.OUT_DIR / "eda_delai_x_acompte.csv")
    print("\nTaux d'annulation — délai x acompte :")
    print(delai_tab)

    _figure_eda(eda, delai_tab)

    # ----------------------------------------------------------------- #
    # 3. Protocole de validation temporelle
    # ----------------------------------------------------------------- #
    titre("3. PROTOCOLE DE VALIDATION TEMPORELLE")
    folds = D.temporal_folds(train)
    bornes = D.fold_dates(train, folds)
    bornes.to_csv(C.OUT_DIR / "plis_temporels.csv", index=False)
    print(bornes.to_string(index=False))

    # ----------------------------------------------------------------- #
    # 4. Comparaison des familles de modèles
    # ----------------------------------------------------------------- #
    titre("4. COMPARAISON DES MODÈLES (validation temporelle à fenêtre étendue)")
    noms = list(M.build_all())
    records: list[dict] = []
    oof = {n: [] for n in noms + ["Ensemble (final)"]}

    for k, (itr, iva) in enumerate(folds, start=1):
        tr_raw, va_raw = train.iloc[itr], train.iloc[iva]
        reference = F.fit_reference(tr_raw)
        TR = F.engineer(tr_raw, reference)
        VA = F.engineer(va_raw, reference)
        y_tr, y_va = TR[C.TARGET].values, VA[C.TARGET].values

        probas: dict[str, np.ndarray] = {}
        for nom, modele in M.build_all().items():
            cols = (
                C.RAW_NUMERIC if "baseline" in nom else C.MODEL_NUMERIC
            ) + C.CATEGORICAL
            modele.fit(TR[cols], y_tr)
            p = modele.predict_proba(VA[cols])[:, 1]
            probas[nom] = p
            seuil, f1 = E.best_threshold(y_va, p)
            m = E.metrics(y_va, p, seuil)
            records.append({"pli": k, "modele": nom, **m})
            oof[nom].append(pd.DataFrame({"y": y_va, "proba": p, "pli": k}))

        p_ens = M.ensemble_proba(
            {n: probas[n] for n in M.ENSEMBLE_MEMBERS if n in probas}
        )
        seuil, _ = E.best_threshold(y_va, p_ens)
        records.append(
            {"pli": k, "modele": "Ensemble (final)", **E.metrics(y_va, p_ens, seuil)}
        )
        oof["Ensemble (final)"].append(
            pd.DataFrame({"y": y_va, "proba": p_ens, "pli": k})
        )
        print(f"  pli {k}/{len(folds)} terminé ({len(itr)} train / {len(iva)} val)")

    detail = pd.DataFrame(records)
    detail.to_csv(C.OUT_DIR / "resultats_par_pli.csv", index=False)
    resume = E.summarize_folds(records)
    resume.to_csv(C.OUT_DIR / "resultats_resume.csv", index=False)
    print("\nMoyenne sur les 4 plis temporels :")
    print(
        resume[["modele", "f1_mean", "f1_std", "precision_mean", "rappel_mean", "roc_auc_mean"]]
        .to_string(index=False)
    )

    # ----------------------------------------------------------------- #
    # 5. Choix du seuil sur les prédictions hors-échantillon agrégées
    # ----------------------------------------------------------------- #
    titre("5. CHOIX DU SEUIL DE DÉCISION")
    oof_ens = pd.concat(oof["Ensemble (final)"], ignore_index=True)
    oof_ens.to_csv(C.OUT_DIR / "oof_ensemble.csv", index=False)
    seuil_final, f1_oof = E.pooled_threshold(oof_ens)
    print(f"seuil retenu (out-of-fold agrégé) : {seuil_final:.3f}  ->  F1 = {f1_oof:.4f}")

    oof_base = pd.concat(oof["Régression logistique — baseline"], ignore_index=True)
    seuil_base, f1_base = E.pooled_threshold(oof_base)
    print(f"rappel baseline : seuil {seuil_base:.3f} -> F1 = {f1_base:.4f}")
    print(f"gain de l'ensemble sur la baseline : {f1_oof - f1_base:+.4f}")

    # Tableau comparatif sur exactement le même jeu de validation
    # (les 4 800 prédictions hors-échantillon des quatre plis).
    comparatif = []
    for nom, parts in oof.items():
        o = pd.concat(parts, ignore_index=True)
        t, _ = E.best_threshold(o.y.values, o.proba.values)
        comparatif.append({"modele": nom, **E.metrics(o.y.values, o.proba.values, t)})
    comparatif = pd.DataFrame(comparatif).round(4)
    comparatif.to_csv(C.OUT_DIR / "comparatif_oof.csv", index=False)
    print("\nComparatif sur les prédictions hors-échantillon agrégées (n = "
          f"{len(oof_ens)}) :")
    print(
        comparatif[["modele", "seuil", "f1", "precision", "rappel", "roc_auc",
                    "pr_auc", "brier", "taux_alerte"]].to_string(index=False)
    )

    couts = E.cout_metier(oof_ens.y.values, oof_ens.proba.values)
    couts.to_csv(C.OUT_DIR / "courbe_cout.csv", index=False)
    _figure_seuil(oof_ens, seuil_final, couts)

    # Repères indispensables pour lire le F1 : le plancher trivial.
    y_oof = oof_ens.y.values
    f1_trivial = f1_score(y_oof, np.ones_like(y_oof))
    acc_finale = (
        (oof_ens.proba.values >= seuil_final).astype(int) == y_oof
    ).mean()
    print(f"F1 de la stratégie « tout annulé » : {f1_trivial:.4f}")
    print(f"accuracy du modèle final : {acc_finale:.4f} "
          f"(accuracy du « tout maintenu » : {1 - y_oof.mean():.4f})")

    # Calibration et déciles de risque : base de la recommandation métier.
    calib = _table_calibration(oof_ens)
    calib.to_csv(C.OUT_DIR / "calibration.csv")
    print("\nCalibration par tranche de probabilité :")
    print(calib.to_string())

    deciles = _table_deciles(oof_ens)
    deciles.to_csv(C.OUT_DIR / "deciles_risque.csv")
    print("\nDéciles de risque :")
    print(deciles.to_string())

    # Contre-épreuve : que donnerait une validation aléatoire ?
    f1_aleatoire = _validation_aleatoire(train)
    print(f"\nContre-épreuve — CV stratifiée aléatoire (5 plis) : F1 = {f1_aleatoire:.4f}")
    print(f"écart avec la validation temporelle : {f1_aleatoire - f1_oof:+.4f}")

    # ----------------------------------------------------------------- #
    # 6. Analyse d'erreurs sur le dernier pli (le plus récent)
    # ----------------------------------------------------------------- #
    titre("6. ANALYSE D'ERREURS ET ÉQUITÉ")
    itr, iva = folds[-1]
    reference = F.fit_reference(train.iloc[itr])
    TR = F.engineer(train.iloc[itr], reference)
    VA = F.engineer(train.iloc[iva], reference)
    cols = C.MODEL_NUMERIC + C.CATEGORICAL
    ens = M.EnsembleAnnulation(threshold=seuil_final).fit(
        TR[cols], TR[C.TARGET].values, reference
    )
    p_va = ens.predict_proba(VA[cols])
    err = E.error_frame(VA, VA[C.TARGET].values, p_va, seuil_final)
    err.to_csv(C.OUT_DIR / "analyse_erreurs_pli4.csv", index=False)
    print(err.type_erreur.value_counts().to_string())

    par_region = E.performance_par_groupe(err, "region_hotel")
    par_region.to_csv(C.OUT_DIR / "performance_par_region.csv", index=False)
    print("\nF1 par région (effectif >= 60) :")
    print(par_region.to_string(index=False))

    par_destination = E.performance_par_groupe(err, "type_destination")
    par_destination.to_csv(C.OUT_DIR / "performance_par_destination.csv", index=False)

    colonnes_erreur = [
        "reservation_id", "proba", "delai_reservation_jours", "type_acompte",
        "tarif_remboursable", "canal_reservation", "segment_client", "nuits",
        "montant_total_eur", "demandes_speciales", "modifications_reservation",
    ]
    fp = err[err.type_erreur == "faux positif"].nlargest(5, "proba")[colonnes_erreur]
    fn = err[err.type_erreur == "faux négatif"].nsmallest(5, "proba")[colonnes_erreur]
    fp.to_csv(C.OUT_DIR / "top5_faux_positifs.csv", index=False)
    fn.to_csv(C.OUT_DIR / "top5_faux_negatifs.csv", index=False)
    print("\n5 faux positifs les plus confiants :")
    print(fp.to_string(index=False))
    print("\n5 faux négatifs les plus confiants :")
    print(fn.to_string(index=False))

    # Interprétation : coefficients de la régression logistique du pli final.
    logreg = M.build_logreg()
    logreg.fit(TR[cols], TR[C.TARGET].values)
    coefs = E.coefficients_logreg(logreg, C.MODEL_NUMERIC, C.CATEGORICAL)
    coefs.to_csv(C.OUT_DIR / "coefficients_logreg.csv", index=False)
    print("\n12 variables les plus influentes (régression logistique) :")
    print(coefs.head(12).to_string(index=False))
    _figure_coefficients(coefs)
    _figure_erreurs(err)

    # ----------------------------------------------------------------- #
    # 7. Modèle final réentraîné sur tout le train + soumission
    # ----------------------------------------------------------------- #
    titre("7. MODÈLE FINAL ET SOUMISSION")
    reference_full = F.fit_reference(train)
    TRAIN = F.engineer(train, reference_full)
    TEST = F.engineer(test, reference_full)
    final = M.EnsembleAnnulation(threshold=seuil_final).fit(
        TRAIN[cols], y, reference_full
    )
    proba_test = final.predict_proba(TEST[cols])
    pred_test = (proba_test >= seuil_final).astype(int)

    submission = pd.DataFrame(
        {
            "reservation_id": test["reservation_id"],
            "probabilite_annulation": np.round(proba_test, 6),
            "reservation_annulee": pred_test,
        }
    )
    submission.to_csv(C.SUBMISSION_PATH, index=False)
    print(f"submission.csv écrit : {submission.shape[0]} lignes, {submission.shape[1]} colonnes")
    print(f"taux d'alerte sur le test : {pred_test.mean():.4f}")
    print(f"probabilité moyenne : {proba_test.mean():.4f}")

    # Vérifications de conformité imposées par le sujet.
    sample = pd.read_csv(C.DATA_DIR / "sample_submission.csv")
    assert len(submission) == 2000, "submission.csv doit contenir 2 000 lignes"
    assert list(submission.columns) == [
        "reservation_id", "probabilite_annulation", "reservation_annulee",
    ], "colonnes non conformes"
    assert (submission.reservation_id.values == sample.reservation_id.values).all(), (
        "l'ordre des identifiants doit être celui du fichier de test"
    )
    assert submission.probabilite_annulation.between(0, 1).all()
    assert submission.reservation_annulee.isin([0, 1]).all()
    print("Toutes les vérifications de conformité sont passées.")

    print(f"\nDurée totale : {time.time() - t0:.1f} s")


# --------------------------------------------------------------------------- #
# Tables annexes
# --------------------------------------------------------------------------- #
def _table_calibration(oof: pd.DataFrame) -> pd.DataFrame:
    """Probabilité annoncée vs fréquence réellement observée."""
    tranches = pd.cut(oof.proba, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    return (
        oof.groupby(tranches, observed=True)
        .agg(n=("y", "size"), proba_moyenne=("proba", "mean"), taux_reel=("y", "mean"))
        .round(3)
    )


def _table_deciles(oof: pd.DataFrame) -> pd.DataFrame:
    """Taux d'annulation et lift par décile de probabilité prédite."""
    d = oof.assign(decile=pd.qcut(oof.proba, 10, labels=False) + 1)
    table = d.groupby("decile").agg(
        n=("y", "size"), proba_moyenne=("proba", "mean"), taux_reel=("y", "mean")
    )
    table["lift"] = (table.taux_reel / oof.y.mean()).round(2)
    table["part_annulations"] = (
        d.groupby("decile")["y"].sum() / oof.y.sum()
    ).round(3)
    return table.round(3)


def _validation_aleatoire(train: pd.DataFrame) -> float:
    """Contre-épreuve : le même modèle évalué en validation croisée aléatoire.

    Sert à vérifier explicitement l'ampleur du biais que le protocole temporel
    permet d'éviter — plutôt que de l'affirmer sans le mesurer.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    cols = C.MODEL_NUMERIC + C.CATEGORICAL
    parts = []
    for itr, iva in skf.split(train, train[C.TARGET]):
        reference = F.fit_reference(train.iloc[itr])
        TR = F.engineer(train.iloc[itr], reference)
        VA = F.engineer(train.iloc[iva], reference)
        probas = {}
        for nom, builder in [
            ("Régression logistique + FE", M.build_logreg),
            ("HistGradientBoosting", M.build_hgb),
            ("LightGBM", M.build_lgbm),
        ]:
            model = builder()
            if model is None:
                continue
            model.fit(TR[cols], TR[C.TARGET].values)
            probas[nom] = model.predict_proba(VA[cols])[:, 1]
        parts.append(
            pd.DataFrame(
                {"y": VA[C.TARGET].values, "proba": M.ensemble_proba(probas)}
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    _, f1 = E.best_threshold(oof.y.values, oof.proba.values)
    return f1


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure_eda(eda: pd.DataFrame, delai_tab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    part = eda[C.TARGET].value_counts(normalize=True).sort_index()
    ax[0, 0].bar(["maintenue", "annulée"], part.values,
                 color=[PALETTE["lagon"], PALETTE["laterite"]])
    for i, v in enumerate(part.values):
        ax[0, 0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
    ax[0, 0].set_title("Déséquilibre de la cible")
    ax[0, 0].set_ylim(0, 0.9)

    for col, color in zip(delai_tab.columns, [PALETTE["laterite"], PALETTE["ocre"], PALETTE["lagon"]]):
        ax[0, 1].plot(range(len(delai_tab)), delai_tab[col], marker="o", label=col, color=color)
    ax[0, 1].set_xticks(range(len(delai_tab)))
    ax[0, 1].set_xticklabels([str(i) for i in delai_tab.index], rotation=20, fontsize=7)
    ax[0, 1].set_title("Effet multiplicatif du délai selon l'acompte")
    ax[0, 1].set_ylabel("taux d'annulation")
    ax[0, 1].legend(fontsize=7, title="acompte")

    serie = (
        eda.set_index("date_reservation")[C.TARGET]
        .resample("QE")
        .mean()
    )
    ax[1, 0].plot(serie.index, serie.values, marker="o", color=PALETTE["laterite"])
    ax[1, 0].axhline(eda[C.TARGET].mean(), ls="--", color=PALETTE["encre"], lw=1)
    ax[1, 0].set_title("Taux d'annulation par trimestre de réservation")
    ax[1, 0].tick_params(axis="x", rotation=30, labelsize=7)

    canal = eda.groupby("canal_reservation")[C.TARGET].mean().sort_values()
    ax[1, 1].barh(canal.index, canal.values, color=PALETTE["lagon"])
    ax[1, 1].set_title("Taux d'annulation par canal de réservation")
    ax[1, 1].tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "01_eda.png", bbox_inches="tight")
    plt.close(fig)


def _figure_seuil(oof: pd.DataFrame, seuil: float, couts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    grid = E.threshold_grid()
    f1s = [f1_score(oof.y, (oof.proba >= t).astype(int)) for t in grid]
    ax[0].plot(grid, f1s, color=PALETTE["laterite"])
    ax[0].axvline(seuil, ls="--", color=PALETTE["lagon"])
    ax[0].set_title(f"F1 selon le seuil (optimum {seuil:.3f})")
    ax[0].set_xlabel("seuil")

    prec, rec, _ = precision_recall_curve(oof.y, oof.proba)
    ax[1].plot(rec, prec, color=PALETTE["lagon"])
    ax[1].axhline(oof.y.mean(), ls="--", color=PALETTE["encre"], lw=1)
    ax[1].set_title("Courbe précision-rappel (out-of-fold)")
    ax[1].set_xlabel("rappel"); ax[1].set_ylabel("précision")

    ax[2].plot(couts.seuil, couts.cout, color=PALETTE["ocre"])
    ax[2].axvline(seuil, ls="--", color=PALETTE["lagon"])
    ax[2].set_title("Coût métier espéré (FN=1, FP=0,25)")
    ax[2].set_xlabel("seuil")

    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "02_seuil.png", bbox_inches="tight")
    plt.close(fig)


def _figure_coefficients(coefs: pd.DataFrame) -> None:
    top = coefs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = [PALETTE["laterite"] if c > 0 else PALETTE["lagon"] for c in top.coefficient]
    ax.barh(top.variable, top.coefficient, color=colors)
    ax.axvline(0, color=PALETTE["encre"], lw=1)
    ax.set_title("15 variables les plus influentes (coefficients standardisés)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "03_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def _figure_erreurs(err: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, color in [(0, PALETTE["lagon"]), (1, PALETTE["laterite"])]:
        ax[0].hist(err.loc[err.y_reel == label, "proba"], bins=30, alpha=0.65,
                   color=color, label="maintenue" if label == 0 else "annulée")
    ax[0].set_title("Distribution des probabilités par classe réelle")
    ax[0].set_xlabel("probabilité prédite"); ax[0].legend(fontsize=8)

    tab = (
        err.groupby("type_erreur", observed=True)["delai_reservation_jours"]
        .median()
        .reindex(["vrai positif", "faux positif", "faux négatif", "vrai négatif"])
    )
    ax[1].bar(tab.index, tab.values,
              color=[PALETTE["lagon"], PALETTE["laterite"], PALETTE["ocre"], PALETTE["encre"]])
    ax[1].set_title("Délai médian de réservation par type d'erreur")
    ax[1].tick_params(axis="x", rotation=15, labelsize=8)

    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "04_erreurs.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
