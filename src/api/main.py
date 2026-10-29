import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictionRequest, LocalRequest, GlobalRequest
from src.model_registry import list_models, load_bundle
from src.explainability import explain_local, explain_global

app = FastAPI(title='BankInsight XAI - Service d’interprétation', version='1.0.0',
    description="Prédiction et explications de la classe positive ; modèles entraînés localement. Voir /api/models pour le schéma réel des variables.",
    docs_url='/api/docs', openapi_url='/api/openapi.json', redoc_url='/api/redoc')


def get_bundle(model_id):
    try:
        return load_bundle(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def validated_frame(bundle, customers):
    train = bundle['X_train']
    expected = set(train.columns)
    for customer in customers:
        if set(customer) != expected:
            raise HTTPException(422, f"Variables attendues exactement : {', '.join(train.columns)}. duration est interdite.")
    frame = pd.DataFrame(customers, columns=train.columns)
    for c in train.columns:
        if pd.api.types.is_numeric_dtype(train[c]):
            try:
                frame[c] = pd.to_numeric(frame[c], errors='raise').astype(float)
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, f"{c} doit être numérique ou null.") from exc
            if np.isinf(frame[c].to_numpy()).any():
                raise HTTPException(422, f"{c} doit être finie.")
        else:
            if any(v is not None and not isinstance(v, str) for v in frame[c]):
                raise HTTPException(422, f"{c} doit être un libellé textuel ou null.")
            frame[c] = frame[c].where(frame[c].notna(), np.nan)
    return frame


@app.get('/api/health', summary='Vérifier le service')
def health():
    return {'status': 'ok', 'message': 'Service disponible'}


@app.get('/api/models', summary='Lister les modèles et leurs variables')
def models():
    return list_models()


@app.post('/api/predict', summary='Prédire la classe positive', responses={200: {'content': {'application/json': {'example': {'probabilities': [.24], 'decisions': [0], 'threshold': .5}}}}})
def predict(request: PredictionRequest):
    bundle = get_bundle(request.model_id)
    frame = validated_frame(bundle, request.customers)
    pipeline = bundle['pipeline']
    p = pipeline.predict_proba(frame)[:, list(pipeline.classes_).index(1)]
    return {'probabilities': p.tolist(), 'decisions': (p >= request.threshold).astype(int).tolist(), 'threshold': request.threshold}


@app.post('/api/explain/local', summary='Expliquer un client avec SHAP et LIME')
def local(request: LocalRequest):
    bundle = get_bundle(request.model_id)
    frame = validated_frame(bundle, [request.customer])
    try:
        return explain_local(bundle, frame.iloc[0].to_dict(), request.threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/explain/global', summary='Expliquer un échantillon du test')
def global_explanation(request: GlobalRequest):
    try:
        return explain_global(get_bundle(request.model_id), request.sample_size)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
