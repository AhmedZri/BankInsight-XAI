"""Registre local de confiance. Aucun import de modèle utilisateur."""
from pathlib import Path
from uuid import uuid4
import json
import os
import re
import joblib
from src.data import ROOT


def registry_dir():
    path = Path(os.environ.get('BANKINSIGHT_REGISTRY', ROOT / 'artifacts' / 'registry'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bundle(result, X_train, X_test, y_test, config):
    model_id = uuid4().hex
    bundle = dict(result, X_train=X_train, X_test=X_test, y_test=y_test, config=config, model_id=model_id)
    path = registry_dir() / model_id
    joblib.dump(bundle, path.with_suffix('.joblib'), compress=3)
    metadata = {k: result[k] for k in ['name', 'cv_ap', 'cv_std', 'params', 'seconds', 'metrics']}
    metadata.update(model_id=model_id, config=config, features=list(X_train.columns))
    path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return model_id


def load_bundle(model_id):
    if not re.fullmatch(r'[a-f0-9]{32}', model_id):
        raise KeyError("Identifiant de modèle invalide.")
    path = registry_dir() / (model_id + '.joblib')
    if not path.exists():
        raise KeyError("Modèle introuvable. Entraîner un modèle ou sélectionner un identifiant existant.")
    return joblib.load(path)


def list_models():
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(registry_dir().glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)]
