# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

## ⚡ Démarrage rapide

Aucune configuration, aucun GPU. L'installation dépend du débit réseau ; le pipeline lui-même s'exécute en **moins d'une minute**.

```bash
# Installation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Exécution
python run_pipeline.py         # 53 s — tables, figures, submission.csv
python ablation.py             # étude d'ablation
jupyter notebook notebook.ipynb   # vue notebook, commentée pas à pas
```

Le pipeline régénère **tous les chiffres, toutes les tables et toutes les figures de ce rapport**, ainsi que `submission.csv`. Il se termine par des vérifications automatiques de conformité du fichier de soumission ; s'il affiche `Toutes les vérifications de conformité sont passées.`, l'exécution est valide.

> Installation détaillée, prérequis et dépannage : **[section 9 — Installation et exécution](#9-installation-et-exécution)**.

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : RANDRIAMIHAJA 
- prénom(s) : Lantoniaina Rojotiana
- classe : ISAIA 4 
- numéro : 17
- rôle : modélisation et pipeline — comparaison des familles de modèles, ensemble pondéré, choix du seuil, industrialisation du code (`src/models.py`, `run_pipeline.py`)

#### Membre 2

- nom : RAKOTO NY AINA
- prénom(s) : Stève Michaël
- classe : IGGLIA 4A 
- numéro : 26
- rôle : exploration et évaluation — EDA, traitement des valeurs manquantes, métriques, analyse d'erreurs et équité par sous-groupes (`src/data.py`, `src/evaluation.py`)

#### Membre 3

- nom : RAKOTOMANGA
- prénom(s) : Titosy Fitia
- classe : IGGLIA 4A 
- numéro : 30
- rôle : feature engineering et protocole de validation — construction des variables, références apprises sur le train, plis temporels à fenêtre étendue, étude d'ablation en huit configurations (`src/features.py`, `ablation.py`)

> **Travail collectif.** La vidéo de présentation, les recommandations métier et la rédaction de ce rapport ont été réalisées **conjointement par les trois membres**. Les rôles ci-dessus indiquent la partie technique dont chacun a assuré la responsabilité principale ; toutes les décisions de modélisation ont été discutées et validées en groupe.

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels perd de la marge sur chaque annulation tardive : une chambre libérée trop près de la date d'arrivée ne se revend plus, et le personnel a déjà été planifié. Sur les 8 000 réservations d'entraînement, **25,84 % ont été annulées** — soit environ une chambre sur quatre à repositionner. L'enjeu n'est pas de détecter l'annulation le jour même mais suffisamment tôt pour agir : relance, incitation à confirmer, ou surréservation maîtrisée. Le modèle doit donc produire une probabilité exploitable, pas seulement une étiquette.

#### Méthodologie adoptée

1. **EDA** — cartographie des manques, contrôle des identités comptables, mise en évidence de la structure multiplicative du risque (délai × conditions commerciales), vérification de la stationnarité du taux d'annulation.
2. **Protocole de validation temporelle d'abord** — quatre plis à fenêtre étendue, construits **avant** toute modélisation, pour éviter de choisir un protocole en fonction du score obtenu.
3. **Baseline imposée** — régression logistique sur les seules variables brutes, tous les prétraitements appris sur le pli d'entraînement.
4. **Feature engineering** mesuré par une étude d'ablation en huit configurations cumulatives (`ablation.py`).
5. **Comparaison de quatre familles** — logistique, forêt aléatoire, gradient boosting histogramme, LightGBM.
6. **Modèle final** — moyenne pondérée des probabilités (0,50 logistique / 0,25 HGB / 0,25 LightGBM), seuil unique choisi sur les 4 800 prédictions hors-échantillon agrégées.
7. **Analyse d'erreurs et équité** — comparaison des faux positifs et faux négatifs, F1 par région et par segment.

#### Résultats obtenus

| Indicateur | Valeur |
|---|---:|
| **F1 (classe « annulation »), hors-échantillon** | **0,4744** |
| F1 de la baseline imposée | 0,4693 |
| Gain du modèle final | **+0,0051** |
| ROC-AUC | 0,673 |
| PR-AUC | 0,392 |
| Score de Brier (calibration) | 0,1777 |
| Seuil retenu | **0,240** |

**Découverte importante.** Le risque d'annulation n'est pas additif : il est **multiplicatif**. Un délai de réservation long pèse très peu quand un acompte total a été versé (10,1 % → 13,7 % d'annulation entre le court et le très long terme) mais fait presque doubler le risque en l'absence d'acompte (28,9 % → 48,6 %). Autrement dit, ce n'est pas le délai qui annule, c'est **le délai sans engagement financier**.

Seconde découverte, moins agréable mais tout aussi importante : **le plafond de séparabilité de ce jeu de données est bas**. Huit configurations de feature engineering et quatre familles de modèles convergent toutes vers une AUC de 0,66–0,68. Le gain total du travail d'ingénierie sur la baseline est de +0,005 F1. Nous le rapportons tel quel plutôt que de le gonfler par une validation aléatoire, qui aurait donné une estimation flatteuse et fausse.

#### Mots-clés

classification binaire · validation temporelle à fenêtre étendue · F1-score · calibration des probabilités · étude d'ablation · seuil de décision · ensemble pondéré · analyse d'erreurs

---

### **3. Contenu du Repository**

```
.
├── README.md                       ce rapport
├── notebook.ipynb                  EDA -> modélisation -> soumission, exécutable de bout en bout
├── run_pipeline.py                 pipeline complet en script (reproduit tous les chiffres du rapport)
├── ablation.py                     étude d'ablation du feature engineering
├── build_notebook.py               génère notebook.ipynb
├── build_script_video.py           génère rapport/script_video.pdf
├── submission.csv                  prédictions sur reservations_test.csv (2 000 lignes)
├── requirements.txt                dépendances (contraintes minimales)
├── requirements-lock.txt           versions exactes de la dernière exécution vérifiée
├── src/
│   ├── config.py                   chemins, graines, listes de colonnes, hyperparamètres
│   ├── data.py                     chargement, nettoyage, plis temporels
│   ├── features.py                 feature engineering + références apprises sur le train
│   ├── models.py                   modèles comparés et ensemble final
│   └── evaluation.py               métriques, seuil, analyse d'erreurs, coût métier
├── figures/                        graphiques produits par le pipeline
├── outputs/                        toutes les tables chiffrées citées ci-dessous
├── rapport/
│   └── script_video.pdf            script minuté de la vidéo de présentation
└── ressources/                     données fournies par le sujet
```

Le fichier `rapport/script_video.pdf` contient le script minuté de la vidéo (651 mots, 4 min 30, six séquences réparties entre trois intervenants), avec les indications de régie et la liste des chiffres à ne pas se tromper à l'oral.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**](14:55
https://drive.google.com/file/d/1qPvkVd-BmKoUrcdXh0yR3tCKrfHe7NRX/view?usp=drivesdk) 
- [Lien vers le dépôt GitHub](https://github.com/rakotonyainastevemichael/Atlantic-Haven-Hotels)
- Sujet original : `ressources/` — données synthétiques ISPM

**Reproduction complète :**

```bash
python3 -m venv .venv && source .venv/bin/activate   # environnement isolé
pip install -r requirements.txt
python run_pipeline.py            # ~55 s : toutes les tables, figures et submission.csv
python ablation.py                # ~40 s : étude d'ablation du feature engineering
jupyter notebook notebook.ipynb   # vue notebook : même démarche, commentée pas à pas
```

Procédure complète, prérequis et dépannage en **[section 9](#9-installation-et-exécution)**.

---

### **4. Résultats de Modélisation**

Tous les modèles sont évalués sur **exactement le même jeu de validation** : les 4 800 prédictions hors-échantillon issues des quatre plis temporels. Le seuil indiqué est celui qui maximise le F1 pour ce modèle.

| Modèle | Paramètres principaux | Seuil | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Régression logistique — baseline | variables brutes, `C=1`, sans FE | 0,205 | 0,4693 | 0,3331 | 0,7941 | 0,6697 |
| Régression logistique + FE | 36 variables, `C=0,05` | 0,225 | 0,4724 | 0,3421 | 0,7632 | **0,6746** |
| Forêt aléatoire | 500 arbres, `min_samples_leaf=30`, `max_features=0,3` | 0,235 | 0,4698 | 0,3416 | 0,7518 | 0,6603 |
| HistGradientBoosting | 300 itér., `lr=0,04`, 8 feuilles, `l2=5` | 0,225 | 0,4733 | 0,3456 | 0,7510 | 0,6616 |
| LightGBM | 400 itér., `lr=0,03`, 7 feuilles, `reg_lambda=10` | 0,215 | 0,4689 | 0,3376 | 0,7673 | 0,6650 |
| **Modèle final — ensemble pondéré** | **0,50 LogReg + 0,25 HGB + 0,25 LGBM** | **0,240** | **0,4744** | **0,3512** | **0,7307** | 0,6730 |

*Table complète : `outputs/comparatif_oof.csv`. Résultats pli par pli : `outputs/resultats_par_pli.csv`.*

Stabilité sur les quatre plis (moyenne ± écart-type du F1) : baseline 0,4736 ± 0,0058 ; ensemble 0,4769 ± 0,0128. Le modèle final gagne en moyenne mais varie davantage — l'écart entre les deux (0,005) reste **inférieur à l'écart-type inter-plis**, et nous le disons explicitement plutôt que de présenter ce gain comme acquis.

**Seuil de décision retenu : 0,240.**

Il n'a pas été choisi pli par pli puis moyenné — cette pratique surestime la performance. Nous avons agrégé les 4 800 prédictions hors-échantillon des quatre plis et cherché **un seul** seuil, celui qui sera effectivement appliqué au jeu de test. La courbe F1-vs-seuil (`figures/02_seuil.png`) est plate entre 0,20 et 0,30 : le choix exact est peu sensible, ce qui est rassurant pour la mise en production.

**Justification du choix du modèle final :**

Le score seul n'aurait pas départagé les candidats — les cinq modèles tiennent dans un intervalle de 0,005 F1. Trois arguments ont pesé :

1. **Complémentarité des erreurs.** La logistique et le boosting se trompent partiellement sur des réservations différentes ; leur moyenne réduit le nombre de faux positifs de 1 804 à 1 659 sans perdre proportionnellement de vrais positifs. C'est le seul gain net observé.
2. **Calibration.** Aucun modèle n'utilise `class_weight="balanced"` : le déséquilibre est traité par le seuil. À F1 identique, cela ramène le score de Brier de 0,225 à **0,178** et rend la colonne `probabilite_annulation` directement interprétable. Vérification par tranche de probabilité : les réservations prédites à 0,25 s'annulent réellement à 25,6 %, celles prédites à 0,08 à 7,3 % (`outputs/calibration.csv`). C'est ce qui autorise un usage opérationnel gradué plutôt qu'un simple oui/non.
3. **Interprétabilité conservée.** La logistique pèse la moitié de l'ensemble ; ses coefficients restent lisibles et sont présentés au métier (`figures/03_coefficients.png`). Un modèle 100 % boosting aurait fait le même score en étant plus difficile à défendre devant un revenue manager.

Nous avons écarté l'ajout d'interactions explicites au modèle : testées (configuration H de l'ablation), elles améliorent l'AUC de 0,0004 mais dégradent le F1 hors-échantillon de 0,0020. Cette décision est documentée dans `src/config.py`.

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l'accuracy pour cette tâche ?**

Parce que l'accuracy récompense ici le silence. Avec 25,84 % d'annulations, un modèle qui prédit systématiquement « maintenue » obtient **74,4 % d'accuracy** sans jamais rendre le moindre service : aucune annulation détectée, aucune chambre repositionnée. Ce score écrase celui de notre modèle final, qui tombe à **58,5 % d'accuracy** au seuil retenu — et qui est pourtant infiniment plus utile. Optimiser l'accuracy sur ce problème reviendrait littéralement à choisir de ne rien prédire.

Le F1 corrige cela en ne comptant que la classe minoritaire, celle qui a une valeur métier. Il oblige à tenir simultanément la précision (ne pas noyer les équipes sous de fausses alertes) et le rappel (ne pas laisser passer les annulations). À titre de repère, prédire « annulée » pour tout le monde donne un F1 de **0,4077** : c'est le plancher réel à battre, pas 0. Notre ensemble atteint 0,4744, soit **+0,067 sur cette référence triviale** — une manière plus honnête de lire le résultat que de le comparer à zéro.

Nous suivons aussi le PR-AUC (0,392), qui résume la qualité du classement indépendamment du seuil, et le score de Brier (0,178) pour la calibration.

#### **Q2. Dans ce contexte, qu'est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

**Définitions.** *Faux positif* : le modèle annonce une annulation, le client vient. L'hôtel a relancé inutilement, peut-être offert un geste commercial, éventuellement surréservé la chambre — au pire, il faut reloger un client qui s'est présenté. *Faux négatif* : le modèle rassure, le client annule. La chambre reste vide, sans possibilité de revente à la dernière minute.

**Réponse nuancée : le faux négatif coûte plus cher, mais le faux positif est plus dangereux.** Le faux négatif a un coût **direct et chiffrable** — le montant total moyen d'une réservation dans nos données est de 1 010 €, dont une part significative est perdue. Le faux positif coûte peu à l'unité : le prix d'un e-mail, ou d'une remise consentie. Un rapport de coût de l'ordre de 4:1 en défaveur du faux négatif est une hypothèse raisonnable, que nous avons codée dans `evaluation.cout_metier()` et qu'il faudra recalibrer avec le revenue manager.

Mais ce raisonnement unitaire cache le vrai risque. Au seuil 0,240, notre modèle alerte sur **53 % des réservations** avec une précision de 35 %. Sur les 4 800 réservations de validation, cela représente 1 659 faux positifs contre 331 faux négatifs. Si chaque alerte déclenche une action humaine, l'équipe reçoit deux fausses alertes pour une vraie et **cessera de les traiter en trois semaines** — un système ignoré ne vaut rien, quel que soit son F1. Et si le faux positif déclenche une surréservation automatique, le coût n'est plus un e-mail mais un client délogé et un avis en ligne durable.

C'est précisément parce que les deux erreurs ne se comparent pas à l'unité que nous ne livrons pas une seule décision binaire mais **trois niveaux d'intervention** (voir Q7) : le seuil optimal au sens du F1 sert de score de référence, pas de déclencheur opérationnel unique.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

Nous avons mesuré chaque groupe par ablation cumulative, avec le protocole de validation identique (`ablation.py`, table complète dans `outputs/ablation_feature_engineering.csv`) :

| Configuration | Variables | F1 | Gain cumulé | ROC-AUC |
|---|---:|---:|---:|---:|
| A. Variables brutes (baseline) | 16 | 0,4691 | — | 0,6706 |
| B. + indicateurs de manque | 20 | 0,4686 | −0,0004 | 0,6731 |
| C. + historique client | 22 | 0,4696 | +0,0005 | **0,6758** |
| D. + composition du séjour | 27 | 0,4691 | 0,0000 | 0,6746 |
| E. + transformations temporelles | 31 | 0,4693 | +0,0002 | 0,6742 |
| F. + prix relatifs | 33 | 0,4704 | +0,0014 | 0,6742 |
| **G. + score d'engagement** | **36** | **0,4724** | **+0,0034** | 0,6746 |
| H. + interactions explicites | 42 | 0,4704 | +0,0014 | 0,6750 |

**Le seul apport réellement mesurable est le groupe « engagement » : +0,0020 de F1 marginal**, le plus gros pas de toute l'étude. Il contient trois variables :

- `score_engagement = acompte_ordinal − tarif_remboursable`, où l'acompte est codé aucun = 0, partiel = 1, total = 2. Cette variable comprime en une échelle unique allant de −1 (aucun acompte + tarif remboursable : 34,6 % d'annulation) à +2 (acompte total non remboursable : 10,4 %). Elle devient **le coefficient le plus fort du modèle final** (−0,476, odds-ratio 0,62 par écart-type).
- `flexibilite_totale`, indicateur binaire du cas extrême aucun acompte + remboursable.
- `a_modifie`, indicateur d'au moins une modification avant extraction — le taux d'annulation passe de 24,8 % (aucune modification) à 49,4 % (trois modifications).

Les **prix relatifs** apportent un second gain plus modeste (+0,0011) : `prix_relatif_hotel` compare le prix payé à la médiane de l'établissement, `prix_relatif_saison` à la médiane région × mois. Ces deux références sont apprises **sur le seul pli d'entraînement**.

L'**historique client** en taux (`taux_annulation_passe = annulations_passees / reservations_passees`) n'améliore pas le F1 mais donne la meilleure AUC de toute l'étude (0,6758) : il classe mieux sans mieux trancher. Il est conservé pour cette raison et parce qu'il est directement actionnable côté CRM.

**Ce qui n'a pas marché, et que nous rapportons quand même.** Les indicateurs de valeurs manquantes dégradent légèrement le F1. Les interactions explicites, pourtant fondées sur un constat d'EDA solide, dégradent le F1 de 0,0020 : le boosting capte déjà ces effets, et la logistique paie le coût des six paramètres supplémentaires. Elles ont été retirées du modèle final. Le gain total du feature engineering est de **+0,0034 F1** — modeste, et nous ne le présenterons pas autrement.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Trois raisons, de la plus évidente à la plus subtile.

**1. Il inverse la flèche du temps.** Le train couvre des réservations créées du 2023-01-01 au 2025-05-24, le test du 2025-05-24 au 2025-12-31. Un `train_test_split` aléatoire entraînerait le modèle sur des réservations de décembre 2024 pour prédire des réservations de mars 2024 — situation qui ne se produira jamais en production, où l'on prédit toujours vers l'avant.

**2. Il fait fuir les statistiques agrégées.** Nos variables `prix_relatif_hotel` et `prix_relatif_saison` reposent sur des médianes par hôtel et par région × mois. Calculées sur un échantillon mélangé, elles incorporeraient l'information de réservations futures dans les variables des réservations passées. La fuite serait invisible dans les scores — elle les améliorerait, ce qui est exactement le piège.

**3. Il masque la dérive.** Une validation aléatoire moyenne implicitement les périodes et rend le modèle aveugle à toute dégradation temporelle.

**Notre protocole : quatre plis à fenêtre étendue** (`data.temporal_folds`). Chaque pli entraîne sur tout le passé disponible et valide sur les 1 200 réservations suivantes — soit 15 % du train à chaque fois :

| Pli | Train | Fin train | n_train | Validation | n_val | Taux d'annulation val. |
|---:|---|---|---:|---|---:|---:|
| 1 | 2023-01-01 → | 2023-12-11 | 3 200 | 2023-12-11 → 2024-04-18 | 1 200 | 26,00 % |
| 2 | 2023-01-01 → | 2024-04-18 | 4 400 | 2024-04-18 → 2024-09-02 | 1 200 | 25,00 % |
| 3 | 2023-01-01 → | 2024-09-02 | 5 600 | 2024-09-02 → 2025-01-10 | 1 200 | 25,00 % |
| 4 | 2023-01-01 → | 2025-01-10 | 6 800 | 2025-01-10 → 2025-05-24 | 1 200 | 26,42 % |

Ce découpage imite la relation train/test du sujet : à chaque pli, la validation est strictement postérieure. Les fichiers étant déjà triés par `date_reservation` (vérifié : la série est monotone croissante), un découpage par position équivaut à un découpage par date.

**Combien cela nous a-t-il coûté ? Nous avons mesuré, et le résultat mérite d'être rapporté honnêtement.** Une validation croisée stratifiée aléatoire à 5 plis, appliquée au même modèle final, donne un F1 de **0,4719** — soit 0,0025 de *moins* que notre estimation temporelle (0,4744). Sur ce jeu de données précis, le protocole aléatoire ne produit donc **pas** l'estimation optimiste redoutée.

L'explication tient à la stationnarité du phénomène : le taux d'annulation oscille entre 22,6 % et 30,2 % selon les trimestres, sans tendance ni rupture (`figures/01_eda.png`). Il n'y a quasiment pas de dérive à masquer, et les statistiques agrégées que nous apprenons (médianes de prix) sont stables dans le temps, donc leur fuite éventuelle ne rapporte rien.

Faut-il en conclure que le protocole temporel était inutile ? Non — et c'est le point important. **Le protocole n'a pas coûté de performance, mais il a acheté une garantie** : nous savons que notre estimation est valide *par construction*, sans dépendre d'une propriété du jeu de données que nous n'aurions constatée qu'après coup. Si le taux d'annulation avait dérivé, l'écart aurait été significatif et le protocole aléatoire nous aurait induits en erreur. On ne choisit pas un protocole de validation en fonction du score qu'il produit.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

Les scénarios ci-dessous décrivent des **circonstances commerciales et comportementales observables**, jamais des populations. Le sujet le rappelle et l'EDA le confirme : l'écart entre régions est faible (23,5 % à 28,4 %) et disparaît largement une fois les conditions commerciales prises en compte.

1. **Absence d'engagement financier, sur réservation très anticipée.** Sans acompte et avec tarif remboursable, le taux d'annulation atteint **34,6 %** ; au-delà de 90 jours de délai et sans acompte, il monte à **48,6 %** — près du double de la moyenne. Le mécanisme est transparent : rien ne coûte au client de garder l'option ouverte.

2. **Réservation modifiée à répétition.** Aucune modification : 24,8 % d'annulation. Une : 27,3 %. Deux : 31,7 %. **Trois : 49,4 %.** Chaque modification est un signal d'un projet de voyage encore instable.

3. **Réservation via un intermédiaire, sans lien direct avec l'établissement.** Plateformes en ligne 30,4 % et agences 27,8 %, contre 21,5 % en direct sur le site et **14,5 % pour le canal entreprise**. Le canal entreprise reste le coefficient négatif le plus fort du modèle (−0,372) : ces réservations sont adossées à un déplacement professionnel déjà validé.

4. **Séjour de groupe sans acompte.** Le segment groupe affiche 30,8 % contre 23,1 % pour les voyages d'affaires. Un groupe agrège plusieurs décisions individuelles ; il suffit qu'une se défasse.

5. **Client peu investi dans son séjour.** Aucune demande spéciale : 28,2 % d'annulation ; trois demandes ou plus : 20,2 %. Demander un lit bébé ou un étage élevé, c'est déjà se projeter dans le séjour.

**Le profil combiné à surveiller** n'est donc pas une région ni un marché : c'est *une réservation flexible (aucun acompte, tarif remboursable), faite à plus de 90 jours, via un intermédiaire*. Ce croisement concerne 165 réservations du train et affiche **55,2 % d'annulation**, soit plus du double de la moyenne. En y ajoutant le critère « déjà modifiée », on tombe à 43 réservations et **60,5 %** — le taux le plus élevé que nous ayons isolé, mais sur un effectif trop faible pour en faire une règle de gestion.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l'entraînement ?**

**Valeurs manquantes.** Cinq variables sont concernées :

| Variable | Manquants (train) | Traitement |
|---|---:|---|
| `agent_id` | 41,9 % | Absence = réservation directe (dictionnaire de données) → modalité `DIRECT` + indicateur `reservation_directe` |
| `enfants` | 4,4 % | Rempli à 0 (modalité majoritaire, sémantiquement neutre) + indicateur |
| `marche_origine` | 2,4 % | Catégorie explicite `Inconnu` — le manque est lui-même une information |
| `prix_moyen_nuit_eur` | 2,4 % | **Reconstruit exactement** (voir ci-dessous) |
| `demandes_speciales` | 2,0 % | Rempli à 0 + indicateur |

La reconstruction du prix est le point le plus intéressant de l'EDA. Nous avons vérifié que l'identité `montant_total = prix × nuits × chambres × (1 − remise/100)` est exacte sur les 7 807 lignes où le prix est présent (ratio observé : 1,000000 ± 0,000006). Les 193 prix manquants sont donc **récupérés sans erreur** par inversion, et non estimés : `prix = montant_total / (nuits × chambres × (1 − remise/100))`. Aucune information n'est perdue et aucune n'est inventée.

**Catégories jamais observées.** Le jeu de test contient une modalité absente du train : `canal_reservation = "assistant_vocal"` (nouveau canal ouvert après la période d'entraînement). Deux protections, redondantes à dessein :

1. `features.engineer()` remappe toute modalité inconnue vers `AUTRE` en s'appuyant sur l'inventaire des catégories mémorisé à l'entraînement ;
2. `OneHotEncoder(handle_unknown="ignore")` produit un vecteur nul plutôt qu'une exception si une modalité échappe au premier filet.

Concrètement, ces réservations sont scorées sur leurs autres caractéristiques, sans plantage et sans hypothèse arbitraire.

**Comment la fuite est évitée.** L'architecture sépare deux catégories de traitement :

- **`data.clean()`** ne contient que des opérations déterministes ligne à ligne (inversion comptable, constantes, indicateurs). Aucun paramètre n'y est appris, donc aucune fuite n'est possible.
- **`features.fit_reference()`** isole tout ce qui nécessite une statistique agrégée — médianes de prix par hôtel, par région × mois, inventaire des catégories. Cette fonction est appelée **exclusivement sur le pli d'entraînement**, à l'intérieur de la boucle de validation, et le dictionnaire produit est appliqué tel quel à la validation puis au test.
- Les imputations et la standardisation scikit-learn sont encapsulées dans un `Pipeline`, donc `fit` sur le train uniquement.
- Aucune référence n'utilise la cible : le risque de fuite par target encoding est **structurellement écarté**, pas seulement évité par prudence.

#### **Q7. Selon vous, quelle action l'hôtel devrait-il entreprendre lorsqu'une réservation en cours présente une forte probabilité d'annulation ?**

**Jamais annuler, jamais pénaliser.** Un modèle à 35 % de précision se trompe deux fois sur trois quand il alerte : agir de façon irréversible sur cette base serait indéfendable, commercialement comme éthiquement.

Nos probabilités étant calibrées (Brier 0,178), elles supportent un usage gradué. Les déciles de risque hors-échantillon (`outputs/deciles_risque.csv`) donnent la base du dispositif :

| Décile de risque | Probabilité moyenne | Taux d'annulation réel | Lift |
|---|---:|---:|---:|
| 1 (plus faible) | 0,076 | 7,1 % | 0,28 |
| 5 | 0,232 | 24,6 % | 0,96 |
| 8 | 0,351 | 38,3 % | 1,50 |
| 10 (plus élevé) | 0,510 | 42,1 % | **1,64** |

**Dispositif proposé — trois niveaux :**

- **Vert — probabilité < 0,20 : 37 % du volume, 13,3 % d'annulation réelle, 19 % des annulations.** Aucune action. Le modèle sert ici surtout à *ne pas* déranger ces clients, ce qui est déjà une économie.
- **Orange — 0,20 à 0,40 : 47 % du volume, 30,0 % d'annulation, 55 % des annulations.** Action à coût nul et non intrusive : e-mail de confirmation à J−14 proposant de personnaliser le séjour (transfert, table, activité). Une demande spéciale supplémentaire est corrélée à un risque plus faible ; l'action utile est donc **d'engager le client dans son séjour**, pas de le surveiller.
- **Rouge — au-delà de 0,40 : 16 % du volume seulement, 41,6 % d'annulation, 25 % des annulations captées.** Appel du service réservation, avec une offre d'arbitrage explicite : passage à un tarif non remboursable contre 10 % de remise, ou option de report gratuit. Le client garde intégralement la main. En parallèle, le revenue manager peut ouvrir une surréservation **prudente** sur les dates concernées — pas plus que le nombre d'annulations attendues sur la période, jamais sur une réservation individuelle.

Ce découpage est plus exigeant que le seuil unique à 0,240, qui alerterait sur 53 % du portefeuille. En concentrant l'action coûteuse sur 16 % du volume, on rend le dispositif tenable pour les équipes tout en captant un quart des annulations.

**Le bon usage des probabilités est agrégé, pas individuel.** Sur 200 réservations classées à 0,40, en attendre environ 80 annulations est fiable ; affirmer que *cette* réservation-ci s'annulera ne l'est pas. C'est la prévision de taux d'occupation, pas le ciblage individuel, qui crée le plus de valeur ici — et c'est aussi l'usage le moins susceptible de nuire à un client.

**Garde-fous recommandés :** aucune action automatique sans validation humaine sur les réservations à fort montant ; plafonnement du nombre de sollicitations par client ; suivi mensuel du taux de faux positifs par région ; réentraînement trimestriel.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Non, et l'écart mérite d'être regardé de près. Sur le pli de validation le plus récent (1 200 réservations, seuil 0,240) :

| Région | Effectif | Taux réel | F1 | Précision | Rappel |
|---|---:|---:|---:|---:|---:|
| Trentino-Alto Adige (montagne) | 111 | 31,5 % | **0,547** | 0,408 | 0,829 |
| Sicilia (insulaire mixte) | 88 | 36,4 % | 0,545 | 0,429 | 0,750 |
| Lombardia (affaires/lacs) | 164 | 26,8 % | 0,504 | 0,386 | 0,727 |
| Lazio (urbaine culturelle) | 187 | 23,5 % | 0,483 | 0,343 | 0,818 |
| Campania (urbaine côtière) | 131 | 25,2 % | 0,473 | 0,338 | 0,788 |
| Puglia (balnéaire rurale) | 73 | 21,9 % | 0,462 | 0,333 | 0,750 |
| Veneto (urbaine littorale) | 161 | 24,2 % | 0,439 | 0,321 | 0,692 |
| Liguria (littorale) | 99 | 28,3 % | 0,416 | 0,327 | 0,571 |
| Toscana (culturelle rurale) | 130 | 22,3 % | **0,340** | 0,246 | 0,552 |

L'écart va de 0,340 à 0,547, soit un rapport de 1 à 1,6.

**Mais cet écart s'explique en grande partie par un artefact mécanique.** Le F1 croît naturellement avec la prévalence : à qualité de classement égale, une région où l'on annule plus donne un meilleur F1. La corrélation entre taux d'annulation réel et F1 par région est nette dans le tableau — Trentino (31,5 %) et Sicilia (36,4 %) sont en tête, Toscana (22,3 %) en queue. Attribuer cet écart au modèle serait une erreur d'interprétation.

**Limite majeure : la taille des sous-groupes.** Les effectifs vont de 73 à 187 réservations, dont seulement 16 à 47 annulations réelles. L'intervalle de confiance à 95 % sur un F1 estimé sur 16 positifs dépasse ±0,15 : **la moitié des écarts observés est indiscernable du bruit d'échantillonnage**. Nous avons volontairement exclu de la table les groupes de moins de 60 observations. Une comparaison rigoureuse exigerait un test de significativité par bootstrap, non réalisé faute de temps — c'est une limite assumée.

Par segment de clientèle, la dispersion est plus faible : loisirs couple 0,521, groupe 0,504, famille 0,469, solo 0,425, affaires 0,404 (`outputs/analyse_erreurs_pli4.csv`). Le modèle est le plus faible sur le segment affaires, qui est aussi le moins risqué (21,4 %) — même effet de prévalence.

**Conséquence opérationnelle :** ne pas appliquer un seuil unique à tous les établissements. Un seuil calibré par région (ou au minimum par niveau de prévalence) égaliserait mieux la charge de travail entre équipes. Nous ne l'avons pas implémenté : avec 73 réservations dans le plus petit groupe, un seuil régional serait ajusté sur du bruit. C'est une piste à rouvrir avec deux ans de données supplémentaires.

#### **Q9. Analyse des erreurs**

Analyse conduite sur le pli de validation le plus récent : 1 200 réservations, 230 vrais positifs, **428 faux positifs**, **87 faux négatifs**, 455 vrais négatifs. Tables complètes : `outputs/top5_faux_positifs.csv` et `outputs/top5_faux_negatifs.csv`.

**Cinq faux positifs les plus confiants** (le modèle criait à l'annulation, le client est venu) :

| ID | Proba | Délai | Acompte | Remb. | Canal | Segment | Montant |
|---|---:|---:|---|---|---|---|---:|
| R001349 | 0,680 | 429 j | aucun | oui | entreprise | groupe | 3 996 € |
| R000207 | 0,628 | 243 j | aucun | oui | plateforme | solo | 912 € |
| R009568 | 0,612 | 120 j | aucun | oui | agence | groupe | 2 965 € |
| R000051 | 0,606 | 402 j | partiel | oui | plateforme | famille | 518 € |
| R005920 | 0,597 | 25 j | aucun | oui | site hôtel | groupe | 1 677 € |

**Cinq faux négatifs les plus confiants** (le modèle rassurait, le client a annulé) :

| ID | Proba | Délai | Acompte | Remb. | Canal | Segment | Montant |
|---|---:|---:|---|---|---|---|---:|
| R009204 | 0,047 | 29 j | total | non | site hôtel | solo | 452 € |
| R001223 | 0,050 | 41 j | total | non | entreprise | loisirs couple | 1 276 € |
| R009144 | 0,077 | 20 j | total | non | entreprise | famille | 1 053 € |
| R001185 | 0,078 | 34 j | total | non | site hôtel | affaires | 423 € |
| R000145 | 0,080 | 22 j | total | oui | site hôtel | loisirs couple | 1 908 € |

**Raisons possibles de ces erreurs.**

Les deux tableaux sont d'une symétrie frappante : **tous les faux positifs cumulent aucun acompte + tarif remboursable ; tous les faux négatifs ont versé un acompte total.** Le modèle ne se trompe pas au hasard — il applique correctement la règle dominante et se fait piéger par les exceptions. Trois mécanismes se combinent :

1. **Le modèle mesure une opportunité, pas une intention.** L'absence d'acompte rend l'annulation *facile*, elle ne la rend pas *souhaitée*. R001349 avait réservé 429 jours à l'avance pour un groupe d'entreprise à 4 000 € : un séminaire planifié de longue date, où la flexibilité tarifaire est une commodité administrative et non un signal d'hésitation. Le modèle lit la structure commerciale et manque le contexte.
2. **Les faux négatifs relèvent d'événements exogènes.** Annuler après avoir versé un acompte total non remboursable, c'est accepter une perte sèche. Ces annulations ont donc des causes que le dataset **ne contient pas** : maladie, imprévu professionnel, problème de transport. Aucune variable disponible ne peut les prédire — c'est une limite de données, pas de modèle.
3. **Le plafond de séparabilité est atteint.** L'AUC plafonne à 0,67 quelle que soit la famille de modèle. Une part importante du phénomène est simplement irréductible avec les variables fournies.

**Pistes d'amélioration.**

*Sur les données — le levier principal.* Trois ajouts changeraient l'ordre de grandeur du résultat :

- **L'historique d'interaction avec la réservation** : ouvertures d'e-mails, connexions à l'espace client, consultations de la page d'annulation. Un client qui consulte les conditions d'annulation à J−20 est un signal d'intention, pas de structure tarifaire. C'est ce qui manque le plus.
- **Le contexte externe** : météo prévue, événements concurrents, prix des vols vers la destination. Cela expliquerait une partie des faux négatifs.
- **L'horodatage des modifications** : nous savons *combien* de modifications ont eu lieu, pas *quand*. Trois modifications à J−90 et trois à J−5 n'ont pas le même sens.

*Sur le modèle.* Un seuil différencié par niveau de prévalence (Q8) ; un modèle séquentiel réévaluant la probabilité à mesure que la date approche, plutôt qu'une prédiction unique à la réservation ; et un test de significativité par bootstrap sur les écarts entre sous-groupes.

*Sur la démarche.* La leçon principale de cette journée est méthodologique : dans un problème où toutes les approches convergent vers la même AUC, l'effort marginal est mieux investi dans **la calibration, le protocole de validation et la traduction opérationnelle** que dans la course au dixième de point de F1.

---

### **6. Conclusion et Recommandations**

Le modèle final — un ensemble pondéré de trois estimateurs — atteint un **F1 de 0,4744** sur la classe « annulation », contre 0,4693 pour la baseline logistique imposée et 0,4077 pour la stratégie triviale consistant à tout prédire comme annulé. Il classe correctement 73 % des annulations réelles avec une précision de 35 %, et ses probabilités sont bien calibrées (Brier 0,178) : une réservation annoncée à 25 % de risque s'annule effectivement dans 25,6 % des cas.

Les limites sont réelles et nous préférons les énoncer clairement. L'AUC plafonne à 0,67 : les variables disponibles décrivent la *structure commerciale* de la réservation mais pas le *comportement* du client, et cette information manquante borne mécaniquement la performance. Le gain apporté par le feature engineering (+0,003) et par l'ensemble (+0,005) est inférieur à l'écart-type entre plis temporels — nous le présentons comme une amélioration probable, pas comme un résultat établi. Enfin, la performance varie du simple au double selon les régions, mais avec des effectifs trop faibles pour distinguer un effet réel du bruit d'échantillonnage.

Le modèle est utilisable dès aujourd'hui pour de la **prévision agrégée de taux d'annulation** et pour de la **priorisation d'actions commerciales douces**. Il ne l'est pas pour une décision automatisée sur une réservation individuelle.

**Recommandation opérationnelle finale :**

Déployer le modèle en **outil de priorisation à trois niveaux** (vert / orange / rouge), et non en système de décision. Concentrer l'effort commercial sur le décile supérieur, où le taux d'annulation atteint 42,1 % contre 25,8 % en moyenne — soit un lift de 1,64. Le seul geste automatisable sans risque est l'e-mail d'engagement à J−14 sur le niveau orange ; toute action à conséquence commerciale (remise, surréservation) doit rester validée par un humain. Piloter la surréservation sur la **somme des probabilités** par date d'arrivée, jamais sur des décisions individuelles.

En parallèle, engager la collecte des trois familles de données identifiées en Q9 — signaux d'interaction client, contexte externe, horodatage des modifications. C'est là, et non dans le réglage d'hyperparamètres, que se trouve le prochain gain significatif.

---

### **7. Reproductibilité**

- **version de Python** : 3.10 minimum ; validé en **3.10.12** et **3.12**
- **principales bibliothèques et versions** : contraintes minimales dans `requirements.txt`, versions exactes de la dernière exécution vérifiée dans `requirements-lock.txt` — numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10.9, lightgbm 4.7.0
- **graine(s) aléatoire(s)** : `SEED = 42`, fixée dans `src/config.py` et propagée à `random`, `numpy`, et à chaque estimateur (`random_state=SEED`)
- **commande ou procédure d'exécution** :
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python run_pipeline.py     # tables, figures et submission.csv
  python ablation.py         # étude d'ablation
  # ou : jupyter notebook notebook.ipynb   (exécution complète depuis un noyau vierge)
  ```
- **durée approximative d'entraînement** : environ 55 s pour le pipeline complet (4 plis × 5 modèles + modèle final), 40 s pour l'ablation, sur un ordinateur portable standard (Intel Core i5, 8 Go RAM). Aucun GPU requis.
- **environnement utilisé** : local, Ubuntu 22.04, CPU uniquement
- **vérifications automatiques** : `run_pipeline.py` se termine par des assertions contrôlant que `submission.csv` contient bien 2 000 lignes, les trois colonnes exigées, l'ordre des identifiants du fichier de test, des probabilités dans [0, 1] et des décisions dans {0, 1}. Le script échoue si l'une de ces conditions n'est pas remplie.
- **LightGBM optionnel** : si la bibliothèque n'est pas installée, `models.py` renormalise automatiquement les poids de l'ensemble sur les deux modèles restants. Le pipeline s'exécute sans erreur, avec un F1 très légèrement inférieur.

---

### **8. Bibliographie**

- **Référence 1** : Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. — Pipelines, `ColumnTransformer`, métriques.
- **Référence 2** : Ke, G. *et al.* (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 30. — Modèle de boosting utilisé dans l'ensemble.
- **Référence 3** : Antonio, N., de Almeida, A., Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. — Référence méthodologique sur la prédiction d'annulation hôtelière ; a orienté le choix des variables d'engagement commercial.
- **Référence 4** : Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3e éd., chapitre sur l'évaluation temporelle. — Fondement du protocole de validation à fenêtre étendue.
- **Référence 5** : Niculescu-Mizil, A., Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. ICML. — Justification du renoncement à `class_weight="balanced"` au profit du réglage du seuil, pour préserver la calibration.
- **Référence 6** : Documentation officielle scikit-learn, sections *Cross-validation of time series data* et *Probability calibration*. — https://scikit-learn.org/stable/

---

### **9. Installation et Exécution**

#### Prérequis

| Élément | Version | Vérification |
|---|---|---|
| Python | ≥ 3.10 | `python3 --version` |
| pip | ≥ 22 | `pip --version` |
| Module `venv` | inclus | `python3 -m venv --help` |
| Espace disque | ~700 Mo | dont ~670 Mo pour l'environnement virtuel |

Aucun GPU, aucune base de données, aucune connexion réseau n'est nécessaire une fois les dépendances installées. Les données sont fournies dans `ressources/` et versionnées avec le dépôt.

Si `venv` est absent sur Debian/Ubuntu : `sudo apt install python3-venv`.

#### Installation pas à pas

```bash
# 1. Récupérer le projet
git clone https://github.com/rakotonyainastevemichael/Atlantic-Haven-Hotels.git
cd Atlantic-Haven-Hotels

# 2. Créer et activer un environnement virtuel isolé
python3 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt      # ou requirements-lock.txt pour les versions exactes

# 4. Vérifier l'installation
python -c "import numpy, pandas, sklearn, matplotlib, lightgbm; print('Environnement prêt')"
```

L'invite du terminal doit afficher `(.venv)` une fois l'environnement activé. **Tant qu'il n'est pas activé, la commande `python` peut ne pas exister** sur les distributions Linux récentes — utiliser `python3`, ou activer l'environnement.

#### Les trois manières d'exécuter le projet

**a) Pipeline complet — la voie recommandée pour le jury**

```bash
python run_pipeline.py
```

Environ 55 secondes. Le script déroule les sept étapes du rapport et affiche chaque résultat au fur et à mesure : chargement et contrôles d'intégrité, EDA, construction des plis temporels, comparaison des cinq modèles, choix du seuil, analyse d'erreurs, modèle final et soumission.

Il produit :

| Sortie | Contenu |
|---|---|
| `submission.csv` | 2 000 prédictions sur `reservations_test.csv` |
| `outputs/*.csv` | toutes les tables chiffrées citées dans ce rapport |
| `figures/*.png` | figures de l'EDA et de l'évaluation |

**b) Étude d'ablation du feature engineering**

```bash
python ablation.py
```

Environ 40 secondes. Reproduit le tableau des huit configurations de la question Q3 et écrit `outputs/ablation_feature_engineering.csv`.

**c) Notebook commenté**

```bash
jupyter notebook notebook.ipynb
```

Puis *Kernel → Restart & Run All*. Même démarche que le pipeline, avec les commentaires d'analyse intercalés. Exécutable de bout en bout depuis un noyau vierge.

#### Comment vérifier que l'exécution est correcte

`run_pipeline.py` se termine par un bloc d'assertions qui contrôle la conformité de `submission.csv` : 2 000 lignes, les trois colonnes exigées, l'ordre des identifiants du fichier de test, des probabilités dans [0, 1] et des décisions dans {0, 1}. **Le script échoue bruyamment si l'une de ces conditions n'est pas remplie.**

Une exécution valide se termine par :

```
submission.csv écrit : 2000 lignes, 3 colonnes
taux d'alerte sur le test : 0.4900
probabilité moyenne : 0.2449
Toutes les vérifications de conformité sont passées.

Durée totale : 53.3 s
```

Les résultats sont **déterministes** (`SEED = 42`) : les chiffres obtenus doivent correspondre exactement à ceux des sections 4 et 5 de ce rapport. Trois repères pour un contrôle rapide :

| Repère | Valeur attendue |
|---|---:|
| F1 hors-échantillon du modèle final | 0,4744 |
| Seuil retenu | 0,240 |
| Coefficient de `score_engagement` | −0,4756 |

#### Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `error: externally-managed-environment` | installation hors environnement virtuel | créer et activer le `.venv` (étape 2) |
| `python : commande introuvable` | environnement non activé | `source .venv/bin/activate`, ou utiliser `python3` |
| `ModuleNotFoundError: No module named 'src'` | mauvais répertoire courant | exécuter depuis la racine du projet |
| `FileNotFoundError` sur `ressources/…` | données absentes | vérifier que `ressources/` contient les quatre CSV du sujet |
| LightGBM ne s'installe pas | binaire indisponible sur la plateforme | **ignorable** : `models.py` renormalise automatiquement les poids de l'ensemble sur les deux modèles restants ; le pipeline s'exécute sans erreur, avec un F1 très légèrement inférieur |
| Chiffres différents de ceux du rapport | versions de bibliothèques divergentes | `pip install -r requirements-lock.txt` |

#### Structure des dépendances

- **`requirements.txt`** — contraintes minimales, lisibles, pour une installation souple.
- **`requirements-lock.txt`** — sortie de `pip freeze` de la dernière exécution vérifiée, pour une reproduction à l'identique.

---

### **Annexe — Outils d'IA générative utilisés**

Un assistant conversationnel (Claude, Anthropic) a été employé comme support de développement : structuration du code en modules, rédaction des docstrings, relecture critique de la démarche méthodologique et mise en forme du rapport. **Toutes les décisions de modélisation, l'ensemble des résultats chiffrés et leur interprétation ont été produits par l'exécution effective du code de ce dépôt** — aucun chiffre de ce rapport n'est repris d'une source externe ou estimé : chacun est reproductible via `run_pipeline.py` et tracé dans `outputs/`.
