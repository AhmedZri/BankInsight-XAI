"""SHAP et LIME en espace des variables métier, sortie probabilité positive."""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from src.preprocessing import original_feature_mapping


class BusinessAdapter:
    """Codes catégoriels réservés aux perturbations ; le modèle reçoit des libellés."""
    def __init__(self, pipeline, train):
        self.pipeline = pipeline
        self.columns = list(train.columns)
        self.numeric = list(train.select_dtypes(include='number').columns)
        self.categories = {}
        for c in self.columns:
            if c not in self.numeric:
                labels = sorted(train[c].dropna().astype(str).unique().tolist())
                for reserved in ['<manquant>', '<inconnu>']:
                    while reserved in labels:
                        reserved = '_' + reserved
                    labels.append(reserved)
                self.categories[c] = labels
        pre = pipeline.named_steps['preprocess']
        self.fills = dict(zip(self.numeric, pre.named_transformers_['numeric'].named_steps['imputer'].statistics_)) if self.numeric else {}

    def encode(self, frame):
        out = np.empty((len(frame), len(self.columns)))
        for i, c in enumerate(self.columns):
            if c in self.numeric:
                out[:, i] = frame[c].astype(float).fillna(self.fills[c])
            else:
                values = self.categories[c]
                out[:, i] = [values.index(str(v)) if pd.notna(v) and str(v) in values
                             else len(values)-2 if pd.isna(v) else len(values)-1 for v in frame[c]]
        return out

    def decode(self, values):
        out = pd.DataFrame(values, columns=self.columns)
        for c, categories in self.categories.items():
            codes = [int(np.clip(round(v), 0, len(categories)-1)) for v in out[c]]
            out[c] = [np.nan if code == len(categories)-2 else categories[code] for code in codes]
        return out

    def proba(self, values):
        return self.pipeline.predict_proba(self.decode(values))

    def positive(self, values):
        return self.proba(values)[:, list(self.pipeline.classes_).index(1)]


def shap_values(pipeline, train, samples, background_size=20, permutations=3):
    adapter = BusinessAdapter(pipeline, train)
    background = adapter.encode(train.sample(min(background_size, len(train)), random_state=42))
    encoded = adapter.encode(samples)
    explainer = shap.Explainer(adapter.positive, shap.maskers.Independent(background),
                              algorithm='permutation', feature_names=adapter.columns, seed=42)
    result = explainer(encoded, max_evals=(2 * len(adapter.columns) + 1) * permutations)
    # Scalar model output guarantees a single positive-class axis, including tree models.
    values = np.asarray(result.values)
    base = np.asarray(result.base_values).reshape(-1)
    predicted = adapter.positive(encoded)
    actual = pipeline.predict_proba(samples)[:, list(pipeline.classes_).index(1)]
    if not np.allclose(predicted, actual, atol=1e-7):
        raise ValueError("Le codage explicatif ne préserve pas la prédiction du client. Vérifier les libellés catégoriels réservés.")
    error = float(np.max(np.abs(base + values.sum(axis=1) - predicted)))
    if values.shape != encoded.shape or error > 1e-5:
        raise ValueError("Échec du contrôle d'additivité SHAP en probabilité positive.")
    return {'values': values.tolist(), 'base_values': base.tolist(), 'data': encoded.tolist(),
            'features': adapter.columns, 'probabilities': predicted.tolist(), 'additivity_error': error,
            'units': 'probabilité de la classe positive', 'background_size': len(background),
            'sample_size': len(samples), 'permutations': permutations,
            'category_labels': adapter.categories}


def explain_local(bundle, customer, threshold=.5, lime_samples=700):
    pipeline, train = bundle['pipeline'], bundle['X_train']
    sample = pd.DataFrame([customer], columns=train.columns)
    sample = sample.where(sample.notna(), np.nan)
    adapter = BusinessAdapter(pipeline, train)
    encoded = adapter.encode(sample)
    result = shap_values(pipeline, train, sample)
    lime = LimeTabularExplainer(adapter.encode(train), feature_names=adapter.columns,
        class_names=['Négative', 'Positive'], categorical_features=[adapter.columns.index(c) for c in adapter.categories],
        categorical_names={adapter.columns.index(c): labels for c, labels in adapter.categories.items()},
        discretize_continuous=True, random_state=42, mode='classification')
    exp = lime.explain_instance(encoded[0], adapter.proba, labels=(1,),
                                num_features=len(adapter.columns), num_samples=lime_samples)
    contributions = dict(exp.local_exp[1])
    lime_by_feature = [float(contributions.get(i, 0)) for i in range(len(adapter.columns))]
    sv = np.asarray(result['values'][0])
    top = np.argsort(np.abs(sv))[::-1][:3]
    phrases = [f"{adapter.columns[i]} ({'hausse' if sv[i] > 0 else 'baisse'} de {abs(sv[i])*100:.1f} points)" for i in top]
    probability = result['probabilities'][0]
    nonzero = [i for i in top if abs(sv[i]) > 1e-8 and abs(lime_by_feature[i]) > 1e-8]
    agreement = float(np.mean([np.sign(sv[i]) == np.sign(lime_by_feature[i]) for i in nonzero])) if nonzero else None
    return {'shap': result, 'probability': probability, 'decision': int(probability >= threshold),
            'threshold': threshold, 'lime': {'rules': exp.as_list(label=1), 'values': lime_by_feature,
                'score': float(exp.score), 'samples': lime_samples, 'local_prediction': float(exp.local_pred[0])},
            'top_sign_agreement': agreement,
            'summary': f"Probabilité de souscription : {probability:.1%}. Les principales associations locales sont : " +
                       ', '.join(phrases) + ". Ces contributions décrivent le modèle et ne démontrent aucun effet causal."}


def explain_global(bundle, sample_size=24):
    pipeline, train = bundle['pipeline'], bundle['X_train']
    test = bundle['X_test']
    sample = test.sample(min(sample_size, len(test)), random_state=42)
    result = shap_values(pipeline, train, sample)
    # Same held-out permutation rows for every model in an experiment.
    subset = test.sample(min(300, len(test)), random_state=42)
    perm = permutation_importance(pipeline, subset, bundle['y_test'].loc[subset.index],
        scoring='average_precision', n_repeats=3, random_state=42, n_jobs=1)
    model = pipeline.named_steps['model']
    native = None
    detail = None
    if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
        weights = model.coef_[0] if hasattr(model, 'coef_') else model.feature_importances_
        mapping = original_feature_mapping(pipeline.named_steps['preprocess'])
        aggregate = pd.Series(np.abs(weights)).groupby(mapping, sort=False).sum()
        native = {c: float(aggregate.get(c, 0)) for c in train.columns}
        detail = [{'encoded_feature': n, 'original_feature': c, 'value': float(w)}
                  for n, c, w in zip(pipeline.named_steps['preprocess'].get_feature_names_out(), mapping, weights)]
    result.update({'permutation': perm.importances_mean.tolist(), 'permutation_std': perm.importances_std.tolist(),
        'permutation_rows': len(subset), 'native': native, 'native_detail': detail,
        'native_units': 'coefficient en log-odds (numériques standardisées si activé)' if hasattr(model, 'coef_') else 'importance par impureté / gain XGBoost',
        'ranking': sorted(zip(train.columns, np.abs(result['values']).mean(axis=0).tolist()), key=lambda t: -t[1])})
    return result
