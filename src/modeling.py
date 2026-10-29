from time import perf_counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from src.preprocessing import make_preprocessor
from src.evaluation import evaluate

MODEL_NAMES = {'logistic': 'Régression logistique', 'tree': 'Arbre de décision',
    'knn': 'K plus proches voisins', 'forest': 'Forêt aléatoire', 'xgboost': 'XGBoost'}


def make_model(name, seed, imbalance_ratio):
    return {
        'logistic': LogisticRegression(max_iter=1200, class_weight='balanced', random_state=seed),
        'tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight='balanced', random_state=seed),
        'knn': KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=1),
        'forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                    class_weight='balanced', random_state=seed, n_jobs=1),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.08, subsample=.9,
                    colsample_bytree=.9, scale_pos_weight=imbalance_ratio, random_state=seed, n_jobs=1,
                    eval_metric='logloss', tree_method='hist')
    }[name]


SEARCH = {'logistic': {'model__C': [.1, 1., 10.]},
    'tree': {'model__max_depth': [3, 6, 10], 'model__min_samples_leaf': [5, 15, 30]},
    'knn': {'model__n_neighbors': [5, 15, 25], 'model__weights': ['uniform', 'distance']},
    'forest': {'model__max_depth': [6, 10, None], 'model__min_samples_leaf': [3, 10]},
    'xgboost': {'model__max_depth': [2, 3, 5], 'model__learning_rate': [.03, .08, .15]}}


def train_models(X_train, X_test, y_train, y_test, names=None, seed=42, search_budget=2,
                 numeric_strategy='median', categorical_strategy='most_frequent', scale=True,
                 overrides=None, progress=None):
    """Toute recherche (préparation comprise) reste à l'intérieur des plis d'entraînement."""
    if any(c.strip().lower() == 'duration' for c in X_train):
        raise ValueError('duration est interdite dans toute chaîne de modélisation.')
    if not 1 <= search_budget <= 9:
        raise ValueError("Budget de recherche : 1 à 9 candidats.")
    results = []
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    for name in names or MODEL_NAMES:
        start = perf_counter()
        model = make_model(name, seed, float((y_train == 0).sum() / y_train.sum()))
        model.set_params(**(overrides or {}).get(name, {}))
        pipeline = Pipeline([('preprocess', make_preprocessor(X_train, numeric_strategy, categorical_strategy, scale)), ('model', model)])
        # Explicit interface overrides are fixed, not silently overwritten by search.
        grid = {k: v for k, v in SEARCH[name].items() if k.removeprefix('model__') not in (overrides or {}).get(name, {})}
        if name == 'knn':
            smallest_fold = len(X_train) - (len(X_train) + 2) // 3
            if 'model__n_neighbors' in grid:
                grid['model__n_neighbors'] = [k for k in grid['model__n_neighbors'] if k <= smallest_fold]
            elif model.n_neighbors > smallest_fold:
                raise ValueError(f'KNN : choisir au maximum {smallest_fold} voisins pour ces plis de validation.')
        search = RandomizedSearchCV(pipeline, grid or [{}], n_iter=min(search_budget,
            int(np.prod([len(v) for v in grid.values()]))), scoring='average_precision',
            cv=cv, random_state=seed, n_jobs=1, error_score='raise', refit=True)
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        p = fitted.predict_proba(X_test)[:, list(fitted.classes_).index(1)]
        results.append({'name': name, 'pipeline': fitted, 'cv_ap': float(search.best_score_),
            'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
            'params': search.best_params_, 'seconds': perf_counter() - start,
            'probabilities': p, 'metrics': evaluate(y_test, p)})
        if progress:
            progress(name)
    return results
