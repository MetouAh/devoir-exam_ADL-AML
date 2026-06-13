# AML — Stacking et Blending — Mini-site multi-pages — design clair scientifique

Projet pédagogique interactif centré sur **Advanced Machine Learning**.

## Démarrage

Ouvrir :

```text
index.html
```

Puis naviguer entre les pages :

```text
pages/01_motivation.html
pages/02_stacking_oof.html
pages/03_blending_comparison.html
pages/04_math_pseudocode.html
pages/05_implementation.html
pages/06_experiments.html
pages/07_aml_adl_bridge.html
pages/08_questions_references.html
```

## Notebooks

Les notebooks sont prêts pour Google Colab :

```text
notebooks/01_stacking_from_scratch.ipynb
notebooks/02_stacking_sklearn_mlxtend.ipynb
notebooks/03_blending_wine_quality.ipynb
notebooks/04_stability_30_seeds.ipynb
notebooks/05_master_complete_colab.ipynb
```

## Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python src/experiment_runner.py
```

## GitHub / Colab

Après dépôt sur GitHub, remplacer `username` et `repository` dans les liens Colab par vos vrais identifiants.

## Focus du document

Cette version se concentre sur **AML — Stacking et Blending**.  
La partie ADL est utilisée uniquement pour la comparaison imposée : régularisation neuronale, Dropout, BatchNorm, LayerNorm, label smoothing et augmentation.


## Version corrigée

- Palette claire par défaut : blanc, bleu scientifique, cyan, vert doux, violet maîtrisé.
- Suppression de l’ancrage géographique non souhaité.
- Remplacement par des applications scientifiques et industrielles générales.
- Structure multi-pages avec URLs internes.


## Version finale interactive claire

- Couleurs claires par défaut.
- Interface scientifique moderne.
- Pages séparées reliées par URLs internes.
- Suppression de la contextualisation géographique non souhaitée.
- Focus sur Advanced Machine Learning : Stacking, Blending, OOF, Wine Quality et 30 seeds.
- Comparaison ADL uniquement comme pont scientifique obligatoire.


## Améliorations version 90+

- Correction du bouton clair/sombre : le JavaScript utilise maintenant la classe `.dark`, cohérente avec le CSS.
- Ajout d'un protocole deux datasets : Wine Quality UCI + Pima Indians Diabetes avec fallback Breast Cancer.
- Ajout d'un notebook `06_two_datasets_full_protocol.ipynb`.
- Ajout d'un protocole Colab pour générer les résultats réels.
- Ajout de templates `outputs/` pour résultats et stabilité.
- Rapport académique fortement enrichi.
- Questions professeur étendues à 30 questions/réponses.
- Bibliographie enrichie avec références récentes 2025.
- Renforcement mathématique : OOF, risque empirique régularisé, fuite de données, calibration.


## Slides PowerPoint / PDF ajoutées

- `slides/html_powerpoint/presentation_powerpoint_html.html` : version PowerPoint conçue en HTML/CSS.
- `slides/powerpoint/AML_Stacking_Blending_25_slides.pptx` : vraie présentation PowerPoint 25 slides.
- `slides/pdf/AML_Stacking_Blending_25_slides_HTML_CSS.pdf` : PDF exporté depuis le design HTML/CSS.
