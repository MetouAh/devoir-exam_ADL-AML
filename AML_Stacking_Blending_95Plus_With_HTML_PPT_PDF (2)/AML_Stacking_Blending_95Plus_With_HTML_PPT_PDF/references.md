# Bibliographie — Advanced Machine Learning : Stacking et Blending

## Références fondatrices

1. Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks.  
   Contribution : article fondateur du stacking ; introduit l'idée d'apprendre un modèle de second niveau.

2. Breiman, L. (1996). *Stacked Regressions*. Machine Learning.  
   Contribution : renforce l'utilisation de prédictions cross-validées pour entraîner le combinateur.

3. Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3rd ed., O'Reilly.  
   Contribution : référence pédagogique du module AML ; chapitre sur Ensemble Learning.

## Implémentations officielles

4. scikit-learn documentation. *StackingClassifier*.  
   Pertinence : implémentation de production du stacking avec final estimator et prédictions cross-validées.

5. mlxtend documentation. *StackingCVClassifier*.  
   Pertinence : implémentation explicite du stacking avec cross-validation pour préparer les entrées du niveau 2.

6. UCI Machine Learning Repository. *Wine Quality Dataset*.  
   Pertinence : dataset public reconnu ; l'objectif est de modéliser la qualité du vin à partir de propriétés physico-chimiques.

7. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems.  
   Pertinence : publication associée au dataset Wine Quality.

## Modèles de base avancés

8. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.  
   Pertinence : modèle de base et meta-learner possible pour données tabulaires.

9. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.  
   Pertinence : alternative rapide à XGBoost pour grands datasets.

## Pont ADL — régularisation neuronale

10. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.  
    Pertinence : base du lien Dropout ≈ ensemble implicite.

11. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization*. ICML.  
    Pertinence : stabilisation de l'apprentissage profond.

12. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.  
    Pertinence : interprétation bayésienne du dropout et MC Dropout.

## Références récentes ≤ 24 mois

13. Garouani, M., Barhrhouj, A., & Teste, O. (2025). *XStacking: Explanation-Guided Stacked Ensemble Learning*. arXiv.  
    Pertinence : stacking moderne orienté explicabilité ; utile pour l'ouverture récente.

14. Mendes, P. et al. (2025). *Neural Networks Calibration via Learning Uncertainty-Error*. arXiv.  
    Pertinence : discussion récente sur la calibration, centrale pour soft voting et stacking probabiliste.

15. Roschewitz, M. et al. (2025). *Where are we with calibration under dataset shift in image classification?*. arXiv.  
    Pertinence : montre que la calibration doit être évaluée sous changement de distribution.

16. Alvey, J., Contaldi, C. R., & Pieroni, M. (2025). *Simulation-based inference with deep ensembles: Evaluating calibration uncertainty and detecting model misspecification*. arXiv.  
    Pertinence : relie ensembles, incertitude et calibration.
