# Recette réelle à terminer dans un navigateur

## Blocage de la session automatisée

La recette du 28 octobre 2026 a interrogé les surfaces de navigation : `apps=[]`, `browsers=[]`. Après démarrage effectif de FastAPI sur 8020 et Streamlit sur 8520, une ouverture de `http://127.0.0.1:8520` dans Chrome a été tentée. Réponse exacte : **`Browser is not available: chrome`**. Aucun navigateur ni onglet pilotable n’est exposé à cette session ; les captures ne peuvent pas être produites par cet outil. Les figures SHAP exportées dans `assets/figures/` ne sont pas des screenshots.

## Parcours minimal du propriétaire

1. Depuis le dossier du projet, lancer `python scripts/run_local.py --api-port 8020 --ui-port 8520` avec le Python du venv. Si ces ports sont déjà occupés par le service de cette recette, ouvrir directement l’adresse affichée ou choisir deux autres ports libres.
2. Ouvrir `http://127.0.0.1:8520`. Dans **Données**, vérifier 45 211 observations, cible y=yes, aperçu, manquants, distributions, taux positif et corrélations. Capturer `assets/screenshots/01_donnees.png`.
3. Dans **Préparation**, choisir quelques variables, constater l’absence de duration, changer l’imputation ou la standardisation, conserver le split 80/20 et la graine 42, puis appliquer. Capturer `02_preparation.png`.
4. Dans **Modélisation**, sélectionner les cinq modèles, budget 1 ou 2 en mode rapide pour la démonstration, puis entraîner. Vérifier progression, durées, tableau et recommandation par AP CV. Capturer `03_modeles.png`. Pour retrouver le modèle complet livré, utiliser « Parcourir les autres expériences archivées » puis l’expérience avec train 36 168.
5. Dans **Évaluation**, vérifier les cinq courbes ROC/PR, rapport et matrice. Passer le seuil de 0,50 à 0,20 : décisions, rappel/précision/F1 et matrice changent, AP et ROC-AUC restent identiques. Capturer `04_evaluation.png`.
6. Dans **Explainable AI**, choisir le modèle complet, calculer le global puis examiner beeswarm, dépendance, permutation, importance native et dictionnaire des catégories. Capturer `05_shap_global.png`.
7. Choisir le premier client du test puis « Prédire et expliquer ce client ». Vérifier probabilité, waterfall, LIME, synthèse et avertissement de faible fidélité s’il s’affiche. Tester également le formulaire fictif. Capturer `06_shap_lime_local.png` et, si utile, une seconde capture plus bas dans la page.
8. Ouvrir `http://127.0.0.1:8020/api/docs`. Dans `/api/predict`, utiliser les données d’exemple réelles affichées par la section **Service API** (identifiant et toutes les variables), exécuter et constater le code 200. Capturer `07_api.png` sans informations privées.

Créer `assets/screenshots/` si nécessaire. Utiliser la capture native du navigateur ou du système ; ne pas dessiner d’interface factice. Une capture doit montrer le contenu réel après chargement, sans erreur masquée. Les attributs du client UCI livré sont publics ; ne pas capturer un CSV personnel.

## Publication et contrôle public

Le dépôt public cible est désormais `https://github.com/AhmedZri/BankInsight-XAI`, avec identité et droits vérifiés par GitHub CLI. Les 15 captures fournies se trouvent dans `assets/figures/screenshots/` et sont intégrées au README. Aucun nouveau dépôt n’est à créer.

Pour l’hébergement gratuit, suivre le README : Community Cloud, dépôt `AhmedZri/BankInsight-XAI`, branche `main`, fichier **`cloud_app.py`**, Python **3.11**, aucun secret requis. Ce point d’entrée lance la FastAPI canonique en interne puis l’interface originale. Aucune VM ni offre payante n’est nécessaire.

Sur la vraie URL Community Cloud, vérifier les six sections, la probabilité du client archivé et SHAP/LIME. Dans **Service API**, contrôler la santé et télécharger OpenAPI. L’API étant interne, `/api/docs` n’est pas une route publique du domaine Streamlit et `verify_http.py --base URL_STREAMLIT` n’est pas le contrôle adapté. Le script `validate_cloud.py` teste les requêtes HTTP réelles en local ; les interactions distantes doivent ensuite réussir dans l’application publiée. Ajouter la vraie URL au README uniquement après ce contrôle. Sans publication vérifiée, conserver `Final_submission_ready: NO`.
