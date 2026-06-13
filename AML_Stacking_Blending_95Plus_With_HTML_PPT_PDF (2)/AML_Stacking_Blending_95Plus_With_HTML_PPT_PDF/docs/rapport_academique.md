# Note de synthèse académique — AML : Stacking et Blending

## Résumé

Ce rapport présente le chapitre **Advanced Machine Learning — Stacking et Blending** comme une étude rigoureuse du méta-apprentissage appliqué aux méthodes d'ensemble. L'objectif central est de dépasser les limites du vote majoritaire et du soft voting en apprenant une fonction de combinaison à partir des prédictions de plusieurs modèles de base. Le point scientifique le plus important est la construction de la matrice de méta-features par validation croisée **Out-of-Fold**, afin d'éviter la fuite de données. Le protocole expérimental repose principalement sur le dataset **Wine Quality UCI**, avec un dataset secondaire tabulaire pour tester la robustesse du protocole. Les méthodes comparées sont : Best Single Model, Hard Voting, Soft Voting, Custom Stacking, StackingClassifier de scikit-learn, StackingCVClassifier de mlxtend et Custom Blending.

## 1. Introduction

Les méthodes d'ensemble constituent un axe majeur de l'Advanced Machine Learning. Leur principe général consiste à combiner plusieurs prédicteurs pour obtenir un modèle final plus robuste que chaque prédicteur individuel. Le bagging réduit principalement la variance, le boosting réduit progressivement le biais, et le voting combine les sorties par une règle fixe. Le stacking franchit une étape supplémentaire : au lieu de fixer la règle de combinaison, il apprend cette règle à partir des données. Cette idée transforme la question « quel modèle choisir ? » en « comment exploiter plusieurs modèles complémentaires ? ».

Le stacking est particulièrement utile lorsque les modèles de base ont des biais inductifs différents. Une régression logistique modélise des frontières linéaires, une forêt aléatoire capture des interactions non linéaires, un SVM peut représenter des marges complexes, un KNN exploite la géométrie locale, et les modèles de gradient boosting sont très performants sur données tabulaires. L'intérêt du stacking dépend donc de la diversité des erreurs : si tous les modèles commettent les mêmes erreurs, le meta-learner ne dispose pas d'information nouvelle.

## 2. Fondements théoriques

Soit un dataset supervisé D = {(x_i, y_i)} pour i = 1,...,n. On considère M modèles de base h_1, h_2, ..., h_M. Chaque modèle h_j produit une prédiction sur une observation x_i. Ces prédictions sont regroupées dans une matrice Z appelée matrice des meta-features.

Dans le cas de sorties scalaires, on écrit :

Z[i, j] = h_j(x_i)

Dans le cas probabiliste multi-classes :

Z[i, j, c] = P_j(y = c | x_i)

La matrice Z devient l'entrée d'un modèle de second niveau g, appelé meta-learner. La prédiction finale s'écrit :

ŷ = g(Z)

Le meta-learner peut être une régression logistique régularisée, une forêt aléatoire, XGBoost ou LightGBM. Dans un devoir académique, la régression logistique régularisée est un choix de référence solide, car elle limite l'overfitting, offre une interprétation simple des coefficients et permet de comprendre quels base learners contribuent réellement.

## 3. Validation Out-of-Fold et prévention de la fuite

Le principal risque du stacking est la fuite de données. Si l'on entraîne les base learners sur tout le train set puis que l'on utilise leurs prédictions sur ce même train set pour entraîner le meta-learner, les prédictions seront trop optimistes. Certains modèles flexibles peuvent quasiment mémoriser les exemples d'entraînement. Le meta-learner apprend alors une relation artificielle et ne généralise pas.

La procédure Out-of-Fold résout ce problème. Le train set est divisé en K folds. Pour chaque fold k, chaque base learner h_j est entraîné sur D \ D_k et prédit uniquement les exemples du fold D_k. Ainsi, chaque ligne de Z_train est produite par un modèle qui n'a jamais vu l'observation concernée. C'est exactement cette propriété qui rend le stacking statistiquement rigoureux.

Formellement :

h_j^(-k) = train(h_j, D \ D_k)

Pour x_i ∈ D_k :

Z[i, j] = h_j^(-k)(x_i)

Cette construction rend les meta-features comparables à des prédictions de test. Ensuite, les base learners sont réentraînés sur tout le train set pour l'inférence finale, car pour un nouvel exemple x*, on veut exploiter toutes les données disponibles.

## 4. Blending

Le blending est une variante plus simple du stacking. Au lieu d'utiliser K folds, le train set est découpé en deux parties : D_base et D_blend. Les base learners sont entraînés sur D_base, puis prédisent D_blend. Le meta-learner est entraîné sur les prédictions de D_blend.

L'avantage du blending est sa simplicité : il coûte moins cher que le stacking OOF, car chaque base learner est entraîné une seule fois pour produire les meta-features. Sa limite principale est qu'il sacrifie une partie du train set et dépend fortement du choix du hold-out split. Sur un petit dataset, cette instabilité peut être importante.

## 5. Comparaison avec Best Single et Voting

Une comparaison rigoureuse doit inclure trois familles de baselines. La première est le Best Single Model : elle vérifie si un modèle individuel suffit. La deuxième est le VotingClassifier : elle teste si une combinaison fixe améliore les résultats. La troisième est le Stacking/Blending : elle teste si une combinaison apprise apporte un gain.

Le hard voting ignore la confiance des modèles. Le soft voting exploite les probabilités, mais suppose implicitement que les probabilités des base learners sont comparables. Or, un SVM, un KNN et une forêt aléatoire ne sont pas forcément calibrés de la même façon. Le stacking peut apprendre à pondérer implicitement ces sorties, mais il reste sensible à la calibration. C'est pourquoi l'analyse doit discuter la qualité des probabilités et non seulement le score final.

## 6. Protocole expérimental

Le dataset principal est Wine Quality UCI. Les variables sont physico-chimiques : acidité fixe, acidité volatile, acide citrique, sucre résiduel, chlorures, dioxyde de soufre, densité, pH, sulfates et alcool. La cible quality est transformée en classification binaire : quality ≥ 6. Ce choix permet d'utiliser Accuracy, F1 macro, Precision macro, Recall macro et ROC-AUC.

Le protocole comprend :

1. split train/test stratifié ;
2. preprocessing par pipeline scikit-learn ;
3. entraînement des modèles individuels ;
4. Hard Voting et Soft Voting ;
5. CustomStackingClassifier avec OOF ;
6. StackingClassifier scikit-learn ;
7. StackingCVClassifier mlxtend ;
8. CustomBlendingClassifier ;
9. répétition sur 30 graines aléatoires ;
10. sauvegarde des résultats et figures dans outputs/.

## 7. Stabilité sur 30 seeds

Une seule expérience peut donner une conclusion accidentelle. En répétant l'expérience sur 30 graines, on obtient une distribution de la métrique. Il faut comparer la moyenne, l'écart-type, le minimum et le maximum. Un modèle robuste n'est pas seulement celui qui a la meilleure moyenne ; c'est aussi celui qui garde un écart-type faible. Le blending peut parfois obtenir un bon score sur un split particulier, mais être moins stable en moyenne.

## 8. Pourquoi le stacking ne gagne pas toujours

Le stacking ne garantit pas une amélioration automatique. Il peut échouer dans plusieurs cas :

- les base learners sont trop similaires ;
- les probabilités sont mal calibrées ;
- le meta-learner est trop complexe ;
- le dataset est trop petit ;
- la procédure OOF est mal implémentée ;
- le meilleur modèle individuel domine déjà les autres.

Cette discussion est importante en soutenance, car elle montre une compréhension scientifique et non une simple confiance naïve dans les ensembles.

## 9. Lien avec ADL : régularisation neuronale

Le lien avec le chapitre ADL associé se fait principalement via le Dropout. Le stacking combine explicitement plusieurs modèles entraînés séparément. Le Dropout, lui, peut être interprété comme un ensemble implicite de sous-réseaux : à chaque forward pass, un masque aléatoire active une architecture partielle. MC Dropout conserve cette stochasticité à l'inférence pour estimer l'incertitude. Les deux approches cherchent à améliorer la généralisation, mais elles opèrent à des niveaux différents : stacking au niveau des modèles, dropout au niveau des neurones.

BatchNorm et LayerNorm stabilisent les activations, Label Smoothing améliore la calibration des sorties, et Mixup régularise par interpolation de données. La comparaison AML × ADL montre donc que les deux modules traitent le même problème général — généralisation, robustesse et confiance prédictive — avec des outils différents.

## 10. Limites et bonnes pratiques

Les bonnes pratiques principales sont :

- utiliser un pipeline pour éviter la fuite dans le preprocessing ;
- construire Z_train uniquement par OOF ;
- commencer par un meta-learner simple et régularisé ;
- comparer avec Best Single et Voting ;
- répéter sur 30 seeds ;
- analyser les temps d'entraînement ;
- vérifier la calibration des probabilités ;
- sauvegarder tout l'artefact de déploiement : preprocessing, base learners, meta-learner, versions.

## 11. Conclusion

Le stacking est une méthode d'ensemble puissante lorsqu'elle est implémentée avec rigueur. Son apport ne se limite pas à la performance : il introduit une vision méta-apprentissage où le modèle final apprend à exploiter les forces relatives de plusieurs familles d'algorithmes. Le blending est une variante utile pour comprendre le principe, mais le stacking OOF reste plus solide pour une étude académique. Un bon travail de Master doit donc démontrer : la construction OOF, la comparaison expérimentale, la stabilité, le coût et le lien conceptuel avec la régularisation neuronale.
