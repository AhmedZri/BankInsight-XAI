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

Pour l’hébergement, suivre le README mis à jour : le code Docker existant doit être lancé sur une VM gratuite adaptée puis contrôlé sur sa vraie URL. La création d’un Space Docker Hugging Face exige actuellement un abonnement payant et ne satisfait pas la contrainte « aucun service payant ». Le lancement Community Cloud de `app.py` seul ne lance pas FastAPI. Aucun code gelé n’est modifié pour contourner ces limites.

Sur le vrai domaine applicatif du Space, vérifier la page, `/api/docs`, une inférence et SHAP/LIME. Exécuter `python scripts/verify_http.py --base 'URL_APPLICATIVE_REELLE'` depuis le projet local pour le contrôle technique HTTPS/WebSocket. Ajouter ensuite les vraies URL et les captures au README, mettre à jour `docs/conformite.md`, relancer les tests, régénérer le ZIP et vérifier son extraction. Sans ces preuves, conserver `Final_submission_ready: NO`.
