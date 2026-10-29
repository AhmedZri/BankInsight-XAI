import io
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from src import data
from src.data import prepare_data, split_data, read_csv
from src.preprocessing import original_feature_mapping
from src.evaluation import evaluate
from src.model_registry import load_bundle, list_models


def test_csv_and_target(frame):
    csv = frame.to_csv(index=False, sep=';')
    loaded = read_csv(io.StringIO(csv))
    X, y = prepare_data(loaded, 'y', 'yes')
    assert 'duration' not in X
    assert y.equals((loaded.y == 'yes').astype(int))
    assert X.isna().any().any()
    _, inverse = prepare_data(loaded, 'y', 'no')
    assert (y + inverse == 1).all()


@pytest.mark.parametrize('mode', ['one_class', 'missing_target', 'too_small', 'duration', 'no_features', 'unknown_positive'])
def test_invalid_data(frame, mode):
    frame = frame.copy()
    features = None
    positive = 'yes'
    if mode == 'one_class': frame['y'] = 'yes'
    if mode == 'missing_target': frame.loc[0, 'y'] = None
    if mode == 'too_small': frame = frame.iloc[:10]
    if mode == 'duration': features = ['duration']
    if mode == 'no_features': features = []
    if mode == 'unknown_positive': positive = 'inconnue'
    with pytest.raises(ValueError):
        prepare_data(frame, 'y', positive, features)


def test_invalid_csv():
    with pytest.raises(ValueError): read_csv(io.StringIO('a\n'))


def test_deterministic_stratified_split(frame):
    X, y = prepare_data(frame, 'y', 'yes')
    a, b, c, d = split_data(X, y)
    aa, bb, cc, dd = split_data(X, y)
    assert a.equals(aa) and b.equals(bb) and c.equals(cc) and d.equals(dd)
    assert not set(a.index) & set(b.index)
    assert abs(c.mean() - d.mean()) < .04
    a2, b2, _, _ = split_data(X, y, train_limit=60)
    assert len(a2) == 60 and b2.equals(b)


def test_preprocessor_training_only(trained):
    results, a, b, _, _ = trained
    pre = results[0]['pipeline'].named_steps['preprocess']
    expected = a.select_dtypes(include='number').median().to_numpy()
    np.testing.assert_allclose(pre.named_transformers_['numeric'].named_steps['imputer'].statistics_, expected)
    assert pre.named_transformers_['numeric'].named_steps['scaler'].n_samples_seen_ == len(a)
    assert len(original_feature_mapping(pre)) == len(pre.get_feature_names_out())
    unknown = b.iloc[:1].copy()
    unknown['job'] = 'nouvelle catégorie'
    assert np.isfinite(results[0]['pipeline'].predict_proba(unknown)).all()


@pytest.mark.parametrize('name', ['logistic','tree','knn','forest','xgboost'])
def test_all_models(trained, name):
    results, _, b, _, _ = trained
    r = next(r for r in results if r['name'] == name)
    p = r['pipeline'].predict_proba(b)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
    assert 0 <= r['cv_ap'] <= 1


def test_search_never_receives_holdout(frame, monkeypatch):
    import src.modeling as modeling
    from sklearn.model_selection import RandomizedSearchCV
    X, y = prepare_data(frame, 'y', 'yes')
    a, b, c, d = split_data(X, y)
    original = RandomizedSearchCV.fit
    def guarded(self, XX, yy, **kwargs):
        assert set(XX.index) == set(a.index)
        assert not set(XX.index) & set(b.index)
        assert self.scoring == 'average_precision'
        for train, valid in self.cv.split(XX, yy):
            assert not set(train) & set(valid)
        return original(self, XX, yy, **kwargs)
    monkeypatch.setattr(RandomizedSearchCV, 'fit', guarded)
    modeling.train_models(a, b, c, d, names=['logistic'], search_budget=2)


def test_metrics_threshold():
    y, p = [0,0,1,1], [.1,.6,.4,.9]
    low, high = evaluate(y, p, .3), evaluate(y, p, .8)
    assert low['roc_auc'] == high['roc_auc'] == roc_auc_score(y, p)
    assert low['average_precision'] == high['average_precision'] == average_precision_score(y, p)
    assert low['recall'] > high['recall']
    assert low['confusion'] == [[1,1],[0,2]]


def test_fairness_accepts_categorical_age_from_csv():
    from src.evaluation import fairness_groups
    X=pd.DataFrame({'age':['jeune','senior','jeune','senior']})
    groups=fairness_groups(X,[0,1,1,0],[.1,.9,.8,.2],'age')
    assert {g['groupe'] for g in groups}=={'jeune','senior'}
    assert sum(g['effectif'] for g in groups)==4


def test_knn_valid_minimum_csv_and_fixed_neighbors():
    from src.modeling import train_models
    frame = pd.DataFrame({'age': range(30), 'y': ['yes','no'] * 15})
    a,b,c,d = split_data(*prepare_data(frame,'y','yes'), test_size=.4)
    result = train_models(a,b,c,d,names=['knn'],search_budget=2)[0]
    assert result['pipeline'].named_steps['model'].n_neighbors <= 12
    with pytest.raises(ValueError,match='maximum 12'):
        train_models(a,b,c,d,names=['knn'],overrides={'knn':{'n_neighbors':15}})


def test_seeded_model_reproducibility(trained):
    from src.modeling import train_models
    baseline,a,b,c,d=trained
    repeated=train_models(a,b,c,d,names=['forest','xgboost'],search_budget=1)
    for result in repeated:
        previous=next(r for r in baseline if r['name']==result['name'])
        assert previous['params']==result['params']
        np.testing.assert_allclose(previous['probabilities'],result['probabilities'],atol=1e-7)


def test_registry_roundtrip(registry):
    mid, bundle = registry
    assert list_models()[0]['model_id'] == mid
    np.testing.assert_allclose(load_bundle(mid)['probabilities'], bundle['probabilities'])
    with pytest.raises(KeyError): load_bundle('../arbitraire')


def test_verified_cache(tmp_path, monkeypatch):
    cache = tmp_path / 'uci_bank.csv'
    frame = pd.DataFrame({'y': ['yes'] * 5000 + ['no'] * 40211, 'duration': 1})
    frame.to_csv(cache, index=False)
    cache.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}))
    monkeypatch.setattr(data, 'CACHE', cache)
    assert len(data.load_uci()) == 45211
    cache.write_text('altéré')
    with pytest.raises(ValueError, match='altéré'): data.load_uci()


def test_automatic_uci_download(tmp_path, monkeypatch):
    import ucimlrepo
    from types import SimpleNamespace
    frame = pd.DataFrame({'age': range(45211), 'duration': 1})
    y = pd.DataFrame({'y': ['yes'] * 5000 + ['no'] * 40211})
    calls = []
    def fetch(id):
        calls.append(id)
        return SimpleNamespace(data=SimpleNamespace(features=frame, targets=y))
    monkeypatch.setattr(ucimlrepo, 'fetch_ucirepo', fetch)
    monkeypatch.setattr(data, 'CACHE', tmp_path / 'uci.csv')
    assert len(data.load_uci()) == 45211 and calls == [222]
    def unavailable(id): raise OSError('hors ligne')
    monkeypatch.setattr(ucimlrepo, 'fetch_ucirepo', unavailable)
    assert len(data.load_uci(refresh=True)) == 45211
