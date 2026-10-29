import numpy as np
import pandas as pd
import pytest
from src.data import prepare_data, split_data
from src.modeling import train_models
from src.model_registry import save_bundle, load_bundle


@pytest.fixture(scope='session')
def frame():
    rng = np.random.default_rng(9)
    n = 160
    age = rng.integers(18, 80, n).astype(float)
    frame = pd.DataFrame({'age': age, 'job': rng.choice(['admin', 'ouvrier', 'cadre'], n),
        'balance': rng.normal(1000, 800, n), 'duration': rng.integers(1, 400, n),
        'y': np.where(age + rng.normal(0, 20, n) > 55, 'yes', 'no')})
    frame.loc[0:4, 'age'] = np.nan
    frame.loc[10:14, 'job'] = np.nan
    return frame


@pytest.fixture(scope='session')
def trained(frame):
    X, y = prepare_data(frame, 'y', 'yes')
    a, b, c, d = split_data(X, y)
    results = train_models(a, b, c, d, search_budget=1)
    return results, a, b, c, d


@pytest.fixture
def registry(trained, tmp_path, monkeypatch):
    monkeypatch.setenv('BANKINSIGHT_REGISTRY', str(tmp_path / 'registry'))
    results, a, b, _, d = trained
    mid = save_bundle(results[0], a, b, d, {'source': 'synthétique', 'positive':'yes'})
    return mid, load_bundle(mid)
