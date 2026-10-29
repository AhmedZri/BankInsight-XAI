import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api_client import customer_json
from src.explainability import explain_local, shap_values, explain_global


@pytest.mark.parametrize('name', ['logistic','tree','knn','forest','xgboost'])
def test_shap_positive_additivity(trained, name):
    results, a, b, _, _ = trained
    pipeline = next(r['pipeline'] for r in results if r['name'] == name)
    result = shap_values(pipeline, a, b.iloc[:2], background_size=8, permutations=1)
    expected = pipeline.predict_proba(b.iloc[:2])[:, 1]
    np.testing.assert_allclose(result['probabilities'], expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result['base_values']) + np.sum(result['values'],axis=1), expected, atol=1e-6)
    assert result['features'] == list(a.columns)


def test_local_lime_same_customer(registry):
    _, bundle = registry
    customer = customer_json(bundle['X_test'].iloc[0])
    result = explain_local(bundle, customer, lime_samples=350)
    assert result['lime']['rules']
    assert len(result['lime']['values']) == len(bundle['X_train'].columns)
    assert np.isfinite(result['lime']['score'])
    assert result['probability'] == result['shap']['probabilities'][0]
    assert 'causal' in result['summary']


def test_shap_missing_and_unseen_values(registry):
    _, bundle = registry
    sample = bundle['X_test'].iloc[:2].copy()
    sample.loc[sample.index[0], 'age'] = np.nan
    sample.loc[sample.index[0], 'job'] = np.nan
    sample.loc[sample.index[1], 'job'] = 'catégorie inconnue'
    result = shap_values(bundle['pipeline'], bundle['X_train'], sample, background_size=8, permutations=1)
    np.testing.assert_allclose(result['probabilities'], bundle['pipeline'].predict_proba(sample)[:,1], atol=1e-7)


def test_category_labels_do_not_collide_with_missing_marker(frame):
    from src.data import prepare_data,split_data
    from src.modeling import train_models
    frame=frame.copy()
    frame['job']='<manquant>'
    frame.loc[10:20,'job']='<inconnu>'
    a,b,c,d=split_data(*prepare_data(frame,'y','yes'))
    model=train_models(a,b,c,d,names=['logistic'],search_budget=1)[0]['pipeline']
    result=shap_values(model,a,b.iloc[:2],background_size=8,permutations=1)
    np.testing.assert_allclose(result['probabilities'],model.predict_proba(b.iloc[:2])[:,1],atol=1e-7)


def test_direct_local_json_null_matches_api_missing_semantics(registry):
    import pandas as pd
    _,bundle=registry
    customer=customer_json(bundle['X_test'].iloc[0])
    customer['job']=None
    customer['age']=None
    local=explain_local(bundle,customer,lime_samples=350)
    sample=pd.DataFrame([customer],columns=bundle['X_train'].columns)
    sample=sample.where(sample.notna(),np.nan)
    expected=bundle['pipeline'].predict_proba(sample)[0,1]
    assert abs(local['probability']-expected)<1e-7


def test_global_native_mapping(registry):
    _, bundle = registry
    result = explain_global(bundle, sample_size=4)
    assert set(result['native']) == set(bundle['X_train'].columns)
    assert len(result['permutation']) == bundle['X_train'].shape[1]
    assert result['native_detail'] and result['ranking']


def test_api_integration(registry, monkeypatch):
    import src.api_client as api_client
    mid, bundle = registry
    client = TestClient(app)
    assert client.get('/api/health').json()['status'] == 'ok'
    assert client.get('/api/models').json()[0]['model_id'] == mid
    customer = customer_json(bundle['X_test'].iloc[0])
    prediction = client.post('/api/predict', json={'model_id':mid, 'customers':[customer], 'threshold':.4})
    assert prediction.status_code == 200
    p = prediction.json()['probabilities'][0]
    local = client.post('/api/explain/local', json={'model_id':mid, 'customer':customer})
    assert local.status_code == 200, local.text
    assert abs(local.json()['probability'] - p) < 1e-7
    global_response = client.post('/api/explain/global', json={'model_id':mid,'sample_size':3})
    assert global_response.status_code == 200, global_response.text
    assert client.get('/api/docs').status_code == 200
    assert client.get('/api/openapi.json').status_code == 200
    # Exercise the actual frontend client against the real ASGI routes.
    monkeypatch.setattr(api_client.requests, 'post', lambda url, json, timeout: client.post(url.removeprefix('http://127.0.0.1:8000'), json=json))
    assert api_client.api_request('/api/predict', {'model_id':mid,'customers':[customer]})['probabilities'] == [p]


@pytest.mark.parametrize('change', ['unknown_model','missing_feature','duration','bad_numeric','bad_threshold','empty'])
def test_api_validation(registry, change):
    mid, bundle = registry
    customer = customer_json(bundle['X_test'].iloc[0])
    payload = {'model_id':mid, 'customers':[customer]}
    if change == 'unknown_model': payload['model_id'] = '0'*32
    if change == 'missing_feature': customer.pop('job')
    if change == 'duration': customer['duration'] = 42
    if change == 'bad_numeric': customer['age'] = 'abc'
    if change == 'bad_threshold': payload['threshold'] = 1.5
    if change == 'empty': payload['customers'] = []
    assert TestClient(app).post('/api/predict', json=payload).status_code == (404 if change == 'unknown_model' else 422)
