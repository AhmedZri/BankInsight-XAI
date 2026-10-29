"""Expérience reproductible UCI, modèles publics et preuves pour le rapport."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
from time import perf_counter
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load_uci, prepare_data, split_data
from src.modeling import train_models
from src.model_registry import save_bundle, load_bundle
from src.explainability import explain_global, explain_local
from src.api_client import customer_json
from src.evaluation import fairness_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Tout l’entraînement UCI au lieu de 2500 lignes')
    parser.add_argument('--budget', type=int, default=2)
    args = parser.parse_args()
    start = perf_counter()
    df = load_uci()
    X, y = prepare_data(df, 'y', 'yes')
    a, b, c, d = split_data(X, y, train_limit=None if args.full else 2500)
    results = train_models(a, b, c, d, search_budget=args.budget, progress=lambda n: print(n, 'terminé', flush=True))
    config = dict(source='UCI Bank Marketing', target='y', positive='yes', seed=42, test_size=.2,
        train_rows=len(a), test_rows=len(b), train_limit=None if args.full else 2500, search_budget=args.budget,
        numeric_strategy='median', categorical_strategy='most_frequent', scale=True, experiment_id=uuid4().hex)
    config['provenance'] = {
        'dataset_sha256': hashlib.sha256((ROOT / 'data/uci_bank.csv').read_bytes()).hexdigest(),
        'train_index_sha256': hashlib.sha256(a.index.to_numpy(dtype='int64').tobytes()).hexdigest(),
        'test_index_sha256': hashlib.sha256(b.index.to_numpy(dtype='int64').tobytes()).hexdigest(),
        'selection': 'Average Precision CV uniquement ; budget fixé avant cette exécution',
        'holdout_previously_observed': True}
    ids = [save_bundle(r, a, b, d, config) for r in results]
    best = max(range(len(results)), key=lambda i: results[i]['cv_ap'])
    bundle = load_bundle(ids[best])
    print('Sélection CV :', bundle['name'], flush=True)
    global_result = explain_global(bundle)
    # Client pris à une position fixée, sans sélection sur le résultat de la prédiction.
    customer = customer_json(b.iloc[0])
    local_result = explain_local(bundle, customer)
    evidence = dict(config=config, dataset_rows=len(df), positive_rate=float(y.mean()), model_ids=ids,
        selected_id=ids[best], selected_name=bundle['name'], global_explanation=global_result,
        local_explanation=local_result, customer=customer,
        comparison=[{k: r[k] for k in ['name','cv_ap','cv_std','params','seconds','metrics']} for r in results],
        fairness={column: fairness_groups(b, d, bundle['probabilities'], column) for column in ['age','marital']})
    evidence['runtime_seconds'] = perf_counter() - start
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'experiment.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'artifacts' / 'public_demo.json').write_text(json.dumps(ids, indent=2), encoding='utf-8')
    print('Preuves enregistrées : reports/experiment.json', flush=True)


if __name__ == '__main__':
    main()
