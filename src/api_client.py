"""Client HTTP réellement utilisé par l'interface."""
import os
import requests


def api_request(endpoint, payload=None):
    base = os.environ.get('BANKINSIGHT_API_URL', 'http://127.0.0.1:8000').rstrip('/')
    try:
        response = requests.get(base + endpoint, timeout=10) if payload is None else requests.post(base + endpoint, json=payload, timeout=240)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = exc.response.text[:600] if exc.response is not None else str(exc)
        raise RuntimeError("API inaccessible ou requête refusée. Lancer python scripts/run_local.py. " + detail) from exc


def customer_json(row):
    import pandas as pd
    return {c: None if pd.isna(v) else v.item() if hasattr(v, 'item') else v for c, v in row.items()}
