# 30 questions possibles du professeur — AML Stacking et Blending

1. **Pourquoi utiliser Out-of-Fold dans le stacking ?**  
Pour que chaque méta-feature soit produite par un modèle qui n’a jamais vu l’observation correspondante.

2. **Quelle est la différence entre stacking et blending ?**  
Le stacking utilise KFold OOF ; le blending utilise un hold-out set réservé au meta-learner.

3. **Pourquoi le voting peut-il être moins performant ?**  
Il impose une règle fixe alors que le stacking apprend la combinaison.

4. **Pourquoi comparer avec Best Single Model ?**  
Pour vérifier que l’ensemble apporte un gain réel par rapport au meilleur modèle individuel.

5. **Pourquoi comparer avec Hard et Soft Voting ?**  
Pour distinguer l’effet d’une agrégation fixe de l’effet d’une combinaison apprise.

6. **Quel est le risque principal du stacking ?**  
La fuite de données lors de la construction de Z_train.

7. **Pourquoi les prédictions in-sample sont-elles dangereuses ?**  
Elles surestiment la performance des base learners et contaminent le meta-learner.

8. **Pourquoi utiliser predict_proba ?**  
Les probabilités fournissent un signal plus riche que les classes prédites.

9. **Quel problème pose predict_proba ?**  
Les probabilités peuvent être mal calibrées selon les modèles.

10. **Comment traiter la calibration ?**  
Avec CalibratedClassifierCV, Platt scaling, isotonic regression ou une analyse ROC-AUC/ECE.

11. **Quel meta-learner choisir en premier ?**  
Logistic Regression régularisée, car elle est simple, stable et interprétable.

12. **Quand tester XGBoost comme meta-learner ?**  
Si les meta-features sont nombreuses et si la validation montre un gain stable.

13. **Pourquoi la diversité des modèles est-elle importante ?**  
Un ensemble n’améliore que si les modèles font des erreurs différentes.

14. **Stacking gagne-t-il toujours ?**  
Non. Si les base learners sont corrélés ou si le dataset est petit, le gain peut être nul.

15. **Pourquoi faire 30 seeds ?**  
Pour mesurer la stabilité et éviter une conclusion basée sur un split chanceux.

16. **Comment interpréter une boxplot sur 30 seeds ?**  
La médiane, l’IQR et les extrêmes montrent la robustesse du modèle.

17. **Qu’est-ce qu’un bon modèle dans cette étude ?**  
Un modèle avec bonne moyenne F1, faible variance, coût raisonnable et interprétabilité acceptable.

18. **Pourquoi F1 macro ?**  
Parce qu’il donne le même poids aux classes, utile en cas de déséquilibre.

19. **Pourquoi ROC-AUC ?**  
Pour mesurer la capacité de discrimination indépendamment du seuil.

20. **Pourquoi StandardScaler dans un pipeline ?**  
Pour éviter une fuite de statistiques de test vers le train.

21. **Pourquoi StratifiedKFold ?**  
Pour préserver la distribution des classes dans chaque fold.

22. **Pourquoi le blending est-il plus instable ?**  
Il dépend d’un seul split train/blend.

23. **Quand préférer blending ?**  
Quand le dataset est très grand ou quand le coût OOF est prohibitif.

24. **Comment déployer un stacking model ?**  
Sauvegarder preprocessing, base learners réentraînés, meta-learner et versions.

25. **Quel est le lien entre stacking et dropout ?**  
Stacking = ensemble explicite ; dropout = ensemble implicite de sous-réseaux.

26. **MC Dropout sert à quoi ?**  
À estimer l’incertitude prédictive par plusieurs forward passes.

27. **BatchNorm vs LayerNorm ?**  
BatchNorm normalise par mini-batch ; LayerNorm normalise par observation.

28. **Label smoothing et calibration ?**  
Il réduit la sur-confiance des sorties et peut améliorer la calibration.

29. **Que dire si le stacking ne bat pas le meilleur modèle ?**  
C’est un résultat valide : il faut analyser diversité, calibration, coût et stabilité.

30. **Quelle conclusion scientifique défendre ?**  
Le stacking est puissant seulement s’il est construit sans fuite et évalué par des baselines rigoureuses.
