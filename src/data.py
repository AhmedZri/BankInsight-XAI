"""Acquisition, cache vérifié et validation des données."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "uci_bank.csv"
SOURCE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"


def load_uci(refresh=False):
    """Télécharge via ucimlrepo ; replie sur un cache dont le SHA256 est vérifié."""
    def cached():
        manifest = json.loads(CACHE.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(CACHE.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError("Cache UCI altéré : supprimer data/uci_bank.* puis retélécharger.")
        frame = pd.read_csv(CACHE)
        if len(frame) != 45211 or set(frame.y.unique()) != {'yes', 'no'}:
            raise ValueError("Le cache ne correspond pas au jeu UCI attendu.")
        return frame
    if CACHE.exists() and not refresh:
        return cached()
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=222)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        if len(frame) != 45211 or 'duration' not in frame or set(frame.y.unique()) != {'yes', 'no'}:
            raise ValueError("Version UCI inattendue.")
        CACHE.parent.mkdir(exist_ok=True)
        frame.to_csv(CACHE, index=False)
        CACHE.with_suffix('.json').write_text(json.dumps({
            'sha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
            'source': SOURCE, 'rows': len(frame), 'license': 'CC BY 4.0',
            'authors': 'S. Moro, P. Rita et P. Cortez (2014)',
            'doi': '10.24432/C5K306'}, indent=2), encoding='utf-8')
        return frame
    except Exception as exc:
        if CACHE.exists():
            return cached()
        raise RuntimeError("Téléchargement UCI indisponible et aucun cache vérifié. Vérifier Internet puis réessayer.") from exc


def read_csv(source):
    try:
        frame = pd.read_csv(source, sep=None, engine='python')
    except Exception as exc:
        raise ValueError("CSV illisible : utiliser UTF-8 et un séparateur virgule ou point-virgule.") from exc
    if frame.empty or len(frame.columns) < 2 or frame.columns.duplicated().any():
        raise ValueError("Le CSV doit contenir des observations et au moins deux colonnes distinctes.")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Noms de colonnes dupliqués après nettoyage.")
    return frame


def prepare_data(frame, target, positive, features=None):
    if target not in frame or frame[target].isna().any():
        raise ValueError("La cible doit exister et ne contenir aucune valeur manquante.")
    classes = frame[target].unique()
    if len(classes) != 2 or positive not in classes:
        raise ValueError("Choisir une cible binaire et une classe positive existante.")
    y = (frame[target] == positive).astype(int)
    if len(frame) < 30 or y.value_counts().min() < 6:
        raise ValueError("Au moins 30 lignes et 6 exemples par classe sont nécessaires.")
    allowed = [c for c in frame if c != target and c.strip().lower() != 'duration']
    features = allowed if features is None else list(features)
    if not features or len(features) != len(set(features)) or any(c not in allowed for c in features):
        raise ValueError("Variables invalides : duration et la cible sont interdites, sélectionner au moins une variable.")
    X = frame[features].copy().replace([np.inf, -np.inf], np.nan)
    for c in X.select_dtypes(exclude='number'):
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return X, y


def split_data(X, y, test_size=.2, seed=42, train_limit=None):
    if not .1 <= test_size <= .4:
        raise ValueError("La fraction de test doit être comprise entre 10 et 40 %.")
    a, b, c, d = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if train_limit and len(a) > train_limit:
        a, _, c, _ = train_test_split(a, c, train_size=train_limit, stratify=c, random_state=seed)
    if c.value_counts().min() < 3 or d.nunique() != 2:
        raise ValueError("Effectifs insuffisants pour la validation stratifiée à trois plis.")
    return a, b, c, d
