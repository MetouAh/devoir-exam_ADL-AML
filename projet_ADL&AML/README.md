# Pipeline Hybride Deep Learning & Machine Learning

Étude comparative sur deux domaines : **classification de maladies foliaires (PlantVillage)** et **classification de profils de consommation électrique (UCI)**.

## Architecture du pipeline

```
Dataset → Prétraitement → DL (extracteur de features) → Réduction dim. (PCA/UMAP) → ML → Prédiction
```

## Datasets

| Dataset | Tâche | Modèle DL | Réduction | Meilleur ML |
|---|---|---|---|---|
| PlantVillage | Classification maladies (10 classes) | EfficientNetB0 (ImageNet) | PCA 50D | Stacking (RF+LGB+XGB) |
| UCI Électrique | Profils de conso (3 classes) | Conv1D + LSTM | PCA | Stacking (RF+LGB+XGB) |

## Structure du dépôt

```
├── pipeline_hybride_dl_ml.ipynb   # Notebook principal (Google Colab)
├── requirements.txt               # Dépendances Python
├── run_pipeline.py                # Script de lancement
└── README.md
```

## Installation et lancement

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Ou directement sur Google Colab (recommandé — GPU T4) :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 8 Questions Scientifiques traitées

1. **Q1 — Justification du modèle pré-entraîné** : EfficientNetB0 vs ResNet50, DenseNet121, MobileNetV2, ViT-B16
2. **Q2 — Réduction de dimensionnalité** : PCA vs UMAP vs aucune réduction
3. **Q3 — Comparaison des méta-modèles ML** : RF, XGBoost, LightGBM, Stacking
4. **Q4 — Validation croisée stratifiée** : StratifiedKFold(5) sur les deux datasets
5. **Q5 — Interprétabilité** : SHAP TreeExplainer + Permutation Importance
6. **Q6 — Incertitude prédictive** : MC Dropout (PlantVillage) + Deep Ensemble (Électrique)
7. **Q7 — Analyse d'équité (Fairness)** : Sain vs Malade / Jour vs Nuit
8. **Q8 — Coût computationnel** : Temps, RAM, énergie (CO₂)

## Résultats clés

### PlantVillage
| Méthode | Accuracy | F1 |
|---|---|---|
| DL seul (EfficientNetB0) | ~0.85 | ~0.84 |
| ML seul (RF sur pixels) | ~0.55 | ~0.54 |
| **Pipeline Hybride** | **~0.87** | **~0.86** |

### Consommation Électrique
| Méthode | Accuracy | F1 |
|---|---|---|
| DL seul (Conv1D+LSTM) | ~0.75 | ~0.74 |
| ML seul (RF brut) | ~0.70 | ~0.69 |
| **Pipeline Hybride** | **~0.82** | **~0.81** |

## Dépendances principales

- Python 3.10+
- TensorFlow 2.x / Keras
- scikit-learn, XGBoost, LightGBM
- SHAP, UMAP-learn
- Pandas, NumPy, Matplotlib, Seaborn
- CodeCarbon (suivi énergie)

## Reproductibilité

Toutes les graines aléatoires sont fixées (`SEED = 42`) via :
```python
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

## Auteurs
Mariem Mohamed El Bechir - Zahra Yeslick Boubecar - Fatimetou Ahmed Mina - Elkheit Mohamed Babaha
Projet universitaire — ML & DL avancés
Compatible Google Colab T4 GPU.
