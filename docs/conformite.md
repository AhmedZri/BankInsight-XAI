# Conformité au cahier des charges officiel

Autorité : `Projet_Challenge_1_Classification_Explicable (1).pdf`, 4 pages lues intégralement avant l’implémentation substantielle. Les fonctionnalités détaillées supplémentaires du prompt initial sont conservées lorsqu’elles sont compatibles.

| Exigence officielle | Implémentation / preuve |
|---|---|
| p.1 : cas réel et classification explicable | Cas 3, banque/marketing, cible y=yes, données UCI publiques |
| p.2 : téléchargement automatique et CSV | src/data.py, cache SHA256, interface, tests de validation |
| p.2 : aperçu, statistiques, manquants, proportions et exploration | Section 1, distributions Plotly, corrélations, taux positif |
| p.2 : sélection cible/variables, imputation, encodage, normalisation, split | Section 2, Pipeline et ColumnTransformer, tests anti-fuite |
| p.2 : cinq modèles, choix et optimisation des paramètres | Section 3, RandomizedSearchCV 3 plis, deux candidats fixés avant recette ; cinq modèles réentraînés sur 36 168 lignes |
| p.2 : métriques, courbes ROC/PR et seuil interactif | Section 4, evaluate, tests invariance AP/ROC et évolution décisions |
| p.2 : importance native, permutation, beeswarm, dépendance | Section 5, explain_global ; KNN sans importance native |
| p.2 : formulaire ou client test, probabilité, waterfall, LIME, synthèse | Section 5, explain_local, appels HTTP et contrôles positifs/additivité |
| p.3 : analyse critique SHAP/LIME, sensibles, compromis et recommandations | Panel critique, notebook et rapport réel d’une page |
| p.3 : API documentée appelée par l’application | FastAPI, OpenAPI, client HTTP Streamlit, TestClient et smoke HTTP |
| p.3 : notebook et requirements | Notebook de démonstration rapide et lecture de la recette complète via fonctions canoniques ; dépendances directes figées |
| p.3 : application déployée | Docker local validé ; hôte gratuit compatible et vérification HTTP publique restant à obtenir |
| p.3 : dépôt GitHub, liens et captures | Dépôt cible public https://github.com/AhmedZri/BankInsight-XAI ; identité et droits administrateur AhmedZri vérifiés via GitHub CLI ; 15 captures fournies intégrées au README |
| p.3 : PDF exactement une page | reports/one_pager.pdf, résultats de reports/experiment.json ; page comptée et rendu inspecté |
| p.3 : cas 3 et retrait de duration | Interdiction au chargement de la sélection, aucune variable duration dans les modèles |

## Écarts et exigences supplémentaires

Le PDF exige une **application effectivement déployée** et un **dépôt GitHub**. La configuration Docker et l’archive ne remplacent pas un hébergement accessible. Les 15 captures fournies par le propriétaire sont désormais intégrées au README ; elles ne sont pas présentées comme une nouvelle exécution des tests navigateur par l’agent.

Le prompt ajoute tests, ZIP, cache vérifié, contrôle d’additivité, seed, protocoles anti-fuite, informations de classe positive et budget limité. Ces ajouts ne contredisent pas le PDF. Gini/KS concernent le scoring de crédit dans le PDF ; le cas 3 n’en requiert pas. L’AP est utilisée et nommée explicitement comme résumé de PR-AUC en contexte déséquilibré.

## Audit d’acceptation du 28 octobre 2026

Le code de l’interface, de l’API, de la préparation, de la recherche, des métriques et des explications a été relu, avec les tests, artefacts, notebook et rapport. `scripts/audit_release.py` contrôle les artefacts réels : cinq classes de modèles, classe positive 1=yes, absence de duration et de y dans les caractéristiques, transformations ajustées sur les 36 168 lignes d’entraînement, rechargement et probabilités identiques, correspondances des colonnes encodées, invariance AP/ROC-AUC au seuil et additivité SHAP en probabilité. Résultats : `reports/acceptance_audit.json`.

Les recherches restent dans le train par Pipeline et CV stratifiée ; la recommandation est calculée exclusivement à partir de l’AP CV. Les tests instrumentent les indices reçus par RandomizedSearchCV et vérifient l’absence du holdout. Le seed est contrôlé par une répétition forêt/XGBoost sur fixture synthétique. Ces vérifications ne constituent pas une preuve de généralisation prospective.

L’entraînement complet est praticable : **83,14 s**, pic RSS du processus et de ses enfants **476,93 Mio**, deux threads, limites fixées à 900 s et 1 536 Mio. Il conserve exactement le holdout original (SHA256 des indices : `c1c3c7f5f45b5ba6159c45650c2f59b8270c4314b3e403e8a549f9047d292055`). Le holdout avait déjà été consulté : comparaison descriptive rapide/complet, aucune recherche guidée par ces scores. Les données brutes et indices sont également identifiés par empreinte dans `reports/experiment.json`.

Modèle complet retenu par CV : **XGBoost, AP CV 0,427 ; AP test 0,446 ; ROC-AUC test 0,787**. Le modèle rapide retenu était la forêt (AP test 0,398). Le rapport utilise les résultats complets et signale la faible fidélité LIME locale (R² 0,294), les variables sensibles et les limites du protocole.

Corrections ciblées de recette : normalisation cohérente de `null`/`NaN` dans l’explication locale directe ; absence de collision entre libellés CSV réels et marqueurs de catégories manquantes ; limitation du nombre de voisins KNN aux effectifs réels des plis pour les petits CSV ; regroupement d’un âge catégoriel de CSV par libellé au lieu d’un découpage numérique. Ces corrections ont des tests de régression ; aucune architecture ni chaîne ML parallèle n’a été ajoutée.

## Livraison publique et blocages

Le PDF pédagogique original est ignoré par Git et Docker. `scripts/package_submission.py --public` l’exclut de l’archive de publication ; le ZIP privé destiné au professeur le conserve. Seuls les cinq modèles publics UCI approuvés de l’expérience complète sont livrés, sans CSV privés ni autres artefacts locaux.

Mise à jour du 29 octobre 2026 : GitHub CLI est authentifié comme **AhmedZri**, avec accès administrateur au dépôt public indiqué par le propriétaire. Le connecteur GitHub utilise un autre compte et n’est pas employé pour pousser les sources. Le PDF pédagogique et le ZIP privé sont exclus du dépôt public.

La documentation Hugging Face actuelle indique qu’un abonnement payant est requis pour créer un Space Docker, même si CPU Basic n’a pas de coût horaire. Cette option n’est pas retenue sous la contrainte de gratuité. Community Cloud lancé sur `app.py` seul ne démarre pas FastAPI ; le code scientifique et applicatif reste gelé. Le README détaille le déploiement du Docker existant sur une VM gratuite adaptée, ainsi que la commande de vérification HTTP réelle à exécuter une fois l’hôte disponible. Aucun abonnement, service payant ni faux lien applicatif n’a été créé.

Recette Docker issue de l’archive de publication sans PDF : deux CPU, plafond mémoire 2 Gio, pic cgroup 1,48 Gio. Les tests pytest Linux, les endpoints, le WebSocket et le parcours AppTest (préparation appliquée, cinq modèles entraînés depuis l’interface, seuil, SHAP global/local, LIME, formulaire fictif) sont exécutés ; les résultats finaux sont consignés dans `reports/verification.json`. Un arrêt intentionnel de l’API entraîne l’arrêt des services et du conteneur ; sortie 241 due au signal de test, `OOMKilled=false`. Preuves techniques dans `reports/deployment_verification.json`.

Le navigateur de l’agent reste indisponible ; les captures ont été fournies par le propriétaire dans `assets/figures/screenshots/`. AppTest et les tests HTTP ne sont pas présentés comme une recette visuelle. Le parcours complémentaire est défini dans `docs/recette_navigateur.md`. L’hébergement public et la vérification de ses endpoints restent des exigences non satisfaites ; **remise académique finale : NON** tant qu’ils manquent.
