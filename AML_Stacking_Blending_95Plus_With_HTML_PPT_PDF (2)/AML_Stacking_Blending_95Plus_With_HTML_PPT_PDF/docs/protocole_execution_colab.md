# Protocole d'exécution Colab — pour obtenir les outputs réels

## Objectif

Produire les résultats visibles que l'enseignant attend : tableaux, métriques, boxplots, matrices de confusion et fichiers CSV.

## Étapes

1. Ouvrir `notebooks/05_master_complete_colab.ipynb` dans Google Colab.
2. Exécuter toutes les cellules dans l'ordre.
3. Vérifier que le dataset Wine Quality est chargé depuis UCI.
4. Exécuter la comparaison :
   - Best Single
   - Hard Voting
   - Soft Voting
   - Custom Stacking
   - scikit-learn StackingClassifier
   - mlxtend StackingCVClassifier
   - Custom Blending
5. Exécuter l'étude 30 seeds.
6. Télécharger les fichiers du dossier `outputs/`.
7. Remplacer les templates présents dans ce dossier par les résultats générés.

## Fichiers attendus après exécution

```text
outputs/results_table.csv
outputs/stability_30_seeds.csv
outputs/stability_summary.csv
outputs/plots/barplot_f1_macro.png
outputs/plots/boxplot_30_seeds.png
outputs/plots/training_time.png
outputs/plots/confusion_matrix_best.png
outputs/plots/roc_curve_best.png
```

## Note d'honnêteté scientifique

Les fichiers `.csv` fournis dans l'archive sont des templates. Les valeurs finales doivent être produites après exécution du notebook sur Colab.
