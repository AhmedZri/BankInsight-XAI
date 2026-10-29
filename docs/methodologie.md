# Audit scientifique et décisions

## Point de prédiction

L’unité observée est une ligne historique du jeu UCI. Le score est calculé **immédiatement avant l’appel courant planifié**. Il ne sert pas à planifier une campagne avant d’en connaître le calendrier et le canal. Cette précision permet de conserver les variables préalables à cet appel, sous hypothèses explicites.

| Variables | Disponibilité supposée et traitement |
|---|---|
| age, job, marital, education | Profil déjà enregistré dans le CRM ; âge et situation familiale sensibles. Exactitude et actualité à vérifier en production. |
| default, balance, housing, loan | Situation bancaire connue avant l’appel ; aucune donnée de résultat de cet appel. |
| contact, day_of_week (ou day), month | Canal et calendrier de l’appel courant supposés fixés avant sa réalisation. La version ucimlrepo peut nommer le jour day_of_week ; la définition historique UCI décrit le jour du mois. Ne pas lui attribuer une interprétation de jour de semaine sans métadonnées complémentaires. |
| campaign | Rang du contact courant, incluant cet appel ; nombre précédent + 1, supposé connu lors du lancement. Si un CRM ne fournit que le total final de campagne, exclure la variable et réentraîner. |
| pdays, previous, poutcome | Historique de campagnes **précédentes** ; pdays=-1 signifie absence de contact antérieur. Ne pas réécrire ces valeurs avec des résultats futurs. |
| duration | Durée de l’appel, inconnue avant sa fin : exclusion obligatoire dans tous les parcours. |
| y | Résultat à prédire, absent des caractéristiques. |

Aucune autre exclusion automatique n’est justifiée dans ce périmètre conditionnel. Pour un score antérieur au choix de la date, du canal ou au lancement de campagne, retirer les variables alors indisponibles via la sélection et documenter le nouveau protocole. Les noms ne permettent pas de détecter automatiquement toutes les fuites possibles d’un CSV ; son auteur doit auditer la disponibilité des colonnes.

## Validation

Split stratifié 80/20, graine 42 ; test indépendant, avant sous-échantillonnage éventuel du train. CV à trois plis dans le train avec Pipeline complet. Sélection et hyperparamètres par AP CV. L’échantillonnage de démonstration réduit le coût et ne réduit pas le test. Aucune optimisation du seuil n’est formellement réalisée.

Des contacts répétés du même client et une évolution temporelle sont possibles. Sans identifiant client fiable et sans protocole prospectif complet, l’indépendance entre lignes n’est pas garantie. Une validation future temporelle/par client est nécessaire avant généralisation ; le mot « indépendant » décrit ici la séparation des lignes pour le fit, pas une preuve d’indépendance statistique des personnes.

L’AP utilise average_precision_score (somme pondérée des précisions aux incréments de rappel), non l’intégrale trapézoïdale. La ROC-AUC dépend du classement global. Les probabilités des modèles pondérés ne sont pas calibrées.

## Explications

SHAP permutation sur variables originales : références marginales d’entraînement ; catégories redécodées avant le Pipeline, colonnes one-hot jamais perturbées séparément. Fonction expliquée scalaire en probabilité positive pour tous les modèles. La somme des contributions et de la base est contrôlée contre la probabilité réelle ; cela garantit l’additivité, pas l’exactitude des valeurs de Shapley estimées avec trois permutations.

LIME utilise un substitut local régularisé, des catégories discrètes et une discrétisation des continus. Ses coefficients et ceux de la logistique ne se lisent pas comme les contributions de SHAP : la logistique produit des coefficients en log-odds, LIME un poids local conditionné par ses règles, SHAP des contributions additives en probabilité. Les permutations peuvent combiner des profils improbables entre variables corrélées.

Les importances de permutation sont calculées sur le même échantillon de test par expérience, à titre descriptif. Une importance négative indique qu’une permutation a amélioré le score dans cet échantillon, pas un effet causal défavorable. Le nombre de références, clients et perturbations est affiché ; ni les classements globaux sur 24 clients ni un accord local SHAP/LIME ne constituent une preuve de stabilité générale.

Dans les figures SHAP, une catégorie est représentée par un code arbitraire accompagné de son dictionnaire. Une couleur correspondant à un code élevé ne signifie pas que la catégorie est naturellement supérieure à une autre. Les variables numériques conservent leurs valeurs avant standardisation ; l’unité de l’attribution reste la probabilité positive.
