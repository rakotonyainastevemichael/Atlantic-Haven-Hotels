#!/usr/bin/env python3
"""Génère le script de la vidéo de présentation (3 à 5 minutes) en PDF.

Palette « Île Rouge » : latérite, ocre, lagon, crème, encre chaude.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Palette Île Rouge
# --------------------------------------------------------------------------- #
LATERITE = colors.HexColor("#C1440E")
OCRE = colors.HexColor("#E4A11B")
LAGON = colors.HexColor("#0E7C86")
CREME = colors.HexColor("#F7EFE3")
CREME_PALE = colors.HexColor("#FDF9F3")
ENCRE = colors.HexColor("#2B2118")
GRIS = colors.HexColor("#6B5D50")

# --------------------------------------------------------------------------- #
# Polices
# --------------------------------------------------------------------------- #
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DJ", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJ-B", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJ-I", FONT_DIR / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DJ-M", FONT_DIR / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I")

PAGE_W, PAGE_H = A4
MARGE = 18 * mm

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
base = getSampleStyleSheet()

S_TITRE = ParagraphStyle("titre", parent=base["Title"], fontName="DJ-B", fontSize=24,
                         leading=29, textColor=LATERITE, spaceAfter=4)
S_SOUS_TITRE = ParagraphStyle("sst", parent=base["Normal"], fontName="DJ", fontSize=12,
                              leading=16, textColor=GRIS, alignment=TA_CENTER)
S_SECTION = ParagraphStyle("section", parent=base["Normal"], fontName="DJ-B", fontSize=13,
                           leading=17, textColor=colors.white, spaceBefore=0, spaceAfter=0)
S_CORPS = ParagraphStyle("corps", parent=base["Normal"], fontName="DJ", fontSize=10,
                         leading=15.5, textColor=ENCRE, alignment=TA_JUSTIFY,
                         spaceAfter=6)
S_DIRE = ParagraphStyle("dire", parent=S_CORPS, fontSize=10.5, leading=16.5,
                        leftIndent=6, rightIndent=6, spaceAfter=4)
S_REGIE = ParagraphStyle("regie", parent=base["Normal"], fontName="DJ-I", fontSize=8.8,
                         leading=12.5, textColor=LAGON, leftIndent=6, spaceAfter=2)
S_NOTE = ParagraphStyle("note", parent=base["Normal"], fontName="DJ", fontSize=8.8,
                        leading=12.5, textColor=GRIS, alignment=TA_JUSTIFY)
S_PETIT = ParagraphStyle("petit", parent=base["Normal"], fontName="DJ", fontSize=8.5,
                         leading=12, textColor=ENCRE)
S_PETIT_B = ParagraphStyle("petitb", parent=S_PETIT, fontName="DJ-B")
S_H2 = ParagraphStyle("h2", parent=base["Normal"], fontName="DJ-B", fontSize=12,
                      leading=16, textColor=LATERITE, spaceBefore=10, spaceAfter=5)


# --------------------------------------------------------------------------- #
# Habillage des pages
# --------------------------------------------------------------------------- #
def decor(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(LATERITE)
    canvas.rect(0, PAGE_H - 6 * mm, PAGE_W, 6 * mm, stroke=0, fill=1)
    canvas.setFillColor(OCRE)
    canvas.rect(0, PAGE_H - 8 * mm, PAGE_W, 2 * mm, stroke=0, fill=1)

    canvas.setFillColor(GRIS)
    canvas.setFont("DJ", 7.5)
    canvas.drawString(MARGE, 11 * mm,
                      "Atlantic Haven Hotels — script de la vidéo de présentation")
    canvas.drawRightString(PAGE_W - MARGE, 11 * mm, f"page {doc.page}")
    canvas.setStrokeColor(CREME)
    canvas.setLineWidth(0.8)
    canvas.line(MARGE, 14 * mm, PAGE_W - MARGE, 14 * mm)
    canvas.restoreState()


def bandeau(titre, minutage, duree):
    """Bandeau de section coloré avec minutage."""
    t = Table(
        [[Paragraph(titre, S_SECTION),
          Paragraph(f'<font color="#FDF9F3">{minutage}  ·  {duree}</font>',
                    ParagraphStyle("m", parent=S_SECTION, fontSize=9.5,
                                   alignment=2))]],
        colWidths=[112 * mm, 62 * mm],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LATERITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def sequence(titre, minutage, duree, paragraphes, regie=None):
    """Bandeau + bloc de texte, maintenus solidaires sur une même page."""
    return KeepTogether([
        bandeau(titre, minutage, duree),
        Spacer(1, 3 * mm),
        bloc_dire(paragraphes, regie=regie),
        Spacer(1, 5 * mm),
    ])


def bloc_dire(paragraphes, regie=None):
    """Bloc « ce qui est dit » sur fond crème, avec indication de régie."""
    contenu = []
    if regie:
        contenu.append([Paragraph(f"▸ À L'ÉCRAN — {regie}", S_REGIE)])
    for p in paragraphes:
        contenu.append([Paragraph(p, S_DIRE)])
    t = Table(contenu, colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREME_PALE),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, OCRE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def tableau(donnees, largeurs, aligns=None):
    lignes = [[Paragraph(c, S_PETIT_B if i == 0 else S_PETIT) for c in ligne]
              for i, ligne in enumerate(donnees)]
    t = Table(lignes, colWidths=largeurs)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), LAGON),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8CBB8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREME_PALE]),
    ]
    for a in (aligns or []):
        style.append(a)
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Contenu
# --------------------------------------------------------------------------- #
def construire():
    story = []

    # ---------------- Couverture ----------------
    story.append(Spacer(1, 22 * mm))
    story.append(Paragraph("Atlantic Haven Hotels", S_TITRE))
    story.append(Paragraph("Script de la vidéo de présentation — 4 min 30",
                           S_SOUS_TITRE))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "ISPM — M1 Machine Learning &amp; Data Science · Examen final S2",
        S_SOUS_TITRE))
    story.append(Spacer(1, 10 * mm))

    resume = Table([[
        Paragraph(
            "<b>651 mots, soit 4 min 30 à un débit normal (150 mots par "
            "minute) — dans la fourchette 3 à 5 minutes exigée.</b> Les blocs encadrés sont à dire ; les lignes en "
            "italique bleu indiquent ce qui doit être à l'écran. Les chiffres "
            "cités sont ceux produits par <font name='DJ-M' size='8.5'>run_pipeline.py</font> "
            "— ne pas les arrondir différemment à l'oral, un écart avec le "
            "rapport se remarque.", S_CORPS)
    ]], colWidths=[174 * mm])
    resume.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREME),
        ("LINEBEFORE", (0, 0), (0, -1), 3, LATERITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(resume)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Déroulé et minutage", S_H2))
    story.append(tableau([
        ["Séquence", "Minutage", "Durée", "Intervenant"],
        ["1 · L'équipe et le problème", "0:00 – 0:35", "35 s", "Présentateur"],
        ["2 · Ce que disent les données", "0:35 – 1:20", "45 s", "Analyste"],
        ["3 · Protocole de validation", "1:20 – 2:05", "45 s", "Modélisation"],
        ["4 · Baseline et modèle final", "2:05 – 2:55", "50 s", "Modélisation"],
        ["5 · Où le modèle se trompe", "2:55 – 3:40", "45 s", "Analyste"],
        ["6 · Recommandation métier", "3:40 – 4:30", "50 s", "Présentateur"],
    ], [72 * mm, 32 * mm, 24 * mm, 46 * mm]))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Chiffres à ne pas se tromper", S_H2))
    story.append(tableau([
        ["Indicateur", "Valeur", "Où le retrouver"],
        ["F1 du modèle final", "0,4744", "outputs/comparatif_oof.csv"],
        ["F1 de la baseline", "0,4693", "outputs/comparatif_oof.csv"],
        ["F1 « tout annulé »", "0,4077", "sortie de run_pipeline.py"],
        ["Seuil retenu", "0,240", "outputs/oof_ensemble.csv"],
        ["Taux d'annulation", "25,84 %", "EDA"],
        ["Lift du dernier décile", "1,64", "outputs/deciles_risque.csv"],
    ], [52 * mm, 34 * mm, 88 * mm]))

    story.append(PageBreak())

    # ---------------- Séquence 1 ----------------
    story.append(sequence("1 · L'équipe et le problème", "0:00 – 0:35", "35 s", [
        "Bonjour. Nous sommes l'équipe [NOMS], en Master 1 Machine Learning et "
        "Data Science à l'ISPM.",
        "Chez Atlantic Haven Hotels, <b>une réservation sur quatre est annulée</b> — "
        "25,84 % exactement. Quand l'annulation tombe tard, la chambre ne se "
        "revend plus et le personnel a déjà été planifié.",
        "L'enjeu n'est donc pas de constater l'annulation, c'est de la voir venir "
        "assez tôt pour agir. En quatre minutes : ce que disent les données, "
        "comment nous avons validé, et où notre modèle se trompe.",
    ], regie="titre du projet, logo ISPM, noms de l'équipe"))

    # ---------------- Séquence 2 ----------------
    story.append(sequence("2 · Ce que disent les données", "0:35 – 1:20", "45 s", [
        "Deux découvertes ont structuré notre travail.",
        "<b>La première.</b> Le montant total est toujours exactement égal au prix, "
        "fois les nuits, fois les chambres, moins la remise. Nous n'avons donc "
        "pas imputé les 193 prix manquants — nous les avons <b>reconstruits sans "
        "erreur</b>.",
        "<b>La seconde, c'est le cœur du projet.</b> Le risque n'est pas additif, "
        "il est <b>multiplicatif</b>. Avec un acompte total, réserver très à "
        "l'avance ne change presque rien : de 10 à 14 %. Sans acompte, le même "
        "délai fait passer le risque de 29 à 49 %.",
        "<b>Ce n'est pas le délai qui annule, c'est le délai sans engagement "
        "financier.</b> Ce constat nous a donné notre meilleure variable.",
    ], regie="tableau croisé délai × acompte, cases fortes en rouge latérite"))

    story.append(tableau([
        ["Délai de réservation", "Aucun acompte", "Acompte partiel", "Acompte total"],
        ["1 à 7 jours", "28,9 %", "21,1 %", "10,1 %"],
        ["21 à 45 jours", "32,7 %", "22,0 %", "10,2 %"],
        ["Plus de 90 jours", "<b><font color='#C1440E'>48,6 %</font></b>",
         "34,4 %", "13,7 %"],
    ], [50 * mm, 41 * mm, 41 * mm, 42 * mm],
        aligns=[("ALIGN", (1, 0), (-1, -1), "CENTER")]))

    story.append(Spacer(1, 5 * mm))

    story.append(Spacer(1, 5 * mm))

    # ---------------- Séquence 3 ----------------
    story.append(sequence("3 · Le protocole de validation", "1:20 – 2:05", "45 s", [
        "Le jeu de test est plus récent que l'entraînement. Nous avons donc "
        "construit quatre plis à fenêtre étendue : chacun s'entraîne sur tout le "
        "passé et valide sur les 1 200 réservations suivantes. Fixé <b>avant</b> "
        "de voir le moindre score.",
        "Puis nous avons fait la contre-épreuve, et elle nous a surpris : la "
        "validation aléatoire donne 0,4719, soit un peu <b>moins</b> que notre "
        "estimation. Pas de biais optimiste ici — le taux d'annulation est trop "
        "stable pour qu'il y ait quoi que ce soit à masquer.",
        "Le protocole ne nous a rien coûté, mais il nous a acheté une garantie : "
        "notre estimation est valide <b>par construction</b>, pas par chance.",
    ], regie="frise des 4 plis, puis les deux F1 côte à côte : 0,4744 / 0,4719"))

    # ---------------- Séquence 4 ----------------
    story.append(sequence("4 · Baseline et modèle final", "2:05 – 2:55", "50 s", [
        "La baseline imposée donne 0,4693. Nous avons comparé quatre familles de "
        "modèles et mesuré notre feature engineering par une ablation en huit "
        "configurations. La variable la plus utile est celle issue du constat "
        "précédent : le <b>score d'engagement</b>, devenu le premier coefficient "
        "du modèle.",
        "Notre ensemble final atteint <b>0,4744</b>.",
        "Soyons directs : <b>le gain sur la baseline est de cinq millièmes</b>, "
        "inférieur à l'écart-type entre nos plis. L'AUC plafonne à 0,67 quelle que "
        "soit l'approche — les données décrivent la structure commerciale d'une "
        "réservation, pas le comportement du client.",
        "Un choix a vraiment payé : ne pas rééquilibrer les classes. À F1 "
        "identique, le score de Brier passe de 0,225 à <b>0,178</b>. Un client "
        "annoncé à 25 % de risque annule dans 25,6 % des cas.",
    ], regie="tableau comparatif des 5 modèles, puis courbe de calibration"))

    # ---------------- Séquence 5 ----------------
    story.append(sequence("5 · Où le modèle se trompe", "2:55 – 3:40", "45 s", [
        "Nos erreurs sont d'une symétrie frappante.",
        "<b>Tous nos faux positifs</b> — annoncés perdus, mais venus — cumulent "
        "aucun acompte et tarif remboursable. <b>Tous nos faux négatifs</b> "
        "avaient versé un acompte total.",
        "Le modèle ne se trompe donc pas au hasard : il applique la règle "
        "dominante et se fait piéger par les exceptions. Il mesure une "
        "<b>opportunité</b> d'annuler, pas une <b>intention</b>.",
        "Notre plus gros faux positif : un séminaire d'entreprise réservé 429 "
        "jours à l'avance, tarif flexible. Pour le modèle, le profil du "
        "renoncement. En réalité, un événement planifié de longue date.",
        "Ce qui manque n'est pas un meilleur algorithme, c'est une donnée : les "
        "signaux d'interaction du client.",
    ], regie="les deux tableaux d'erreurs côte à côte, colonne acompte surlignée"))

    # ---------------- Séquence 6 ----------------
    story.append(sequence("6 · Recommandation métier", "3:40 – 4:30", "50 s", [
        "Notre recommandation tient en une phrase : <b>ce modèle est un outil de "
        "priorisation, pas un système de décision.</b> À 35 % de précision, il se "
        "trompe deux fois sur trois quand il alerte.",
        "Trois niveaux. <b>Vert</b>, sous 20 % : on ne fait rien. <b>Orange</b> : "
        "un e-mail à J moins 14 proposant de personnaliser le séjour — nos données "
        "montrent qu'un client qui fait une demande spéciale annule moins. "
        "L'action utile est d'<b>engager le client</b>, pas de le surveiller. "
        "<b>Rouge</b>, au-delà de 40 % : un appel avec une offre d'arbitrage. Le "
        "rouge, ce n'est que 16 % du portefeuille.",
        "Et surtout : le bon usage de nos probabilités est <b>agrégé</b>. Prédire "
        "que sur 200 réservations, 80 s'annuleront, c'est fiable. Désigner "
        "laquelle, non.",
        "Merci de votre attention.",
    ], regie="schéma des trois niveaux vert / orange / rouge, puis carte de fin"))

    story.append(Spacer(1, 2 * mm))

    # ---------------- Conseils de tournage ----------------
    story.append(Paragraph("Conseils pour l'enregistrement", S_H2))
    conseils = Table([[
        Paragraph(
            "<b>Enregistrez dès que le modèle final est figé.</b> Le sujet le "
            "rappelle : la vidéo pèse autant que la qualité du code (10 %), et "
            "c'est le livrable le plus souvent sacrifié.<br/><br/>"
            "<b>Répartissez les séquences</b> entre trois voix : cela évite la "
            "monotonie et montre que l'équipe a travaillé ensemble.<br/><br/>"
            "<b>Ne lisez pas.</b> Retenez les six idées-forces et les six "
            "chiffres du tableau de la page 1 ; le reste peut être reformulé. "
            "Le script est calibré au plus juste : chaque phrase ajoutée à "
            "l'oral fait déborder le minutage.<br/><br/>"
            "<b>Assumez le score modeste à l'oral.</b> Un jury distingue "
            "immédiatement une équipe qui annonce 0,47 en expliquant pourquoi le "
            "plafond est là, d'une équipe qui annonce 0,65 obtenu par une "
            "validation aléatoire. La lucidité méthodologique se note.<br/><br/>"
            "<b>Vérifiez le lien</b> avant de le mettre dans le README : partage "
            "public, et durée entre 3 et 5 minutes.",
            S_NOTE)
    ]], colWidths=[174 * mm])
    conseils.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREME),
        ("LINEBEFORE", (0, 0), (0, -1), 3, LAGON),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether(conseils))

    return story


def main():
    sortie = Path(__file__).resolve().parent / "rapport" / "script_video.pdf"
    sortie.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(sortie), pagesize=A4,
        leftMargin=MARGE, rightMargin=MARGE,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Atlantic Haven Hotels — script de la vidéo",
        author="ISPM M1 Machine Learning & Data Science",
    )
    frame = Frame(MARGE, 18 * mm, PAGE_W - 2 * MARGE, PAGE_H - 34 * mm, id="corps")
    doc.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decor)])
    doc.build(construire())
    print(f"PDF généré : {sortie}")


if __name__ == "__main__":
    main()
