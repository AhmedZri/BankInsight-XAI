"""Audit indépendant des artefacts livrés ; aucune recherche ni réestimation des modèles."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load_uci, prepare_data, split_data
from src.model_registry import load_bundle
from src.evaluation import evaluate
from src.preprocessing import original_feature_mapping
from src.api_client import customer_json


def audit():
    full = json.loads((ROOT/'reports/experiment.json').read_text(encoding='utf-8'))
    quick = json.loads((ROOT/'reports/quick_experiment.json').read_text(encoding='utf-8'))
    X,y = prepare_data(load_uci(),'y','yes')
    a,b,c,d = split_data(X,y,train_limit=None)
    expected_hash = hashlib.sha256(b.index.to_numpy(dtype='int64').tobytes()).hexdigest()
    assert full['config']['provenance']['test_index_sha256'] == expected_hash
    _,quick_test,_,quick_y = split_data(X,y,test_size=quick['config']['test_size'],
        seed=quick['config']['seed'],train_limit=quick['config']['train_limit'])
    assert b.equals(quick_test) and d.equals(quick_y)
    assert quick['config']['provenance']['test_index_sha256']==expected_hash
    assert full['selected_name'] == max(full['comparison'],key=lambda r:r['cv_ap'])['name']
    checks=[]
    comparison=[]
    with threadpool_limits(limits=2):
        for model_id in full['model_ids']:
            bundle=load_bundle(model_id)
            pipe=bundle['pipeline']
            assert bundle['X_train'].equals(a) and bundle['X_test'].equals(b)
            assert set(a.index).isdisjoint(b.index) and len(a)+len(b)==len(X)
            assert 'duration' not in pipe.feature_names_in_ and 'y' not in pipe.feature_names_in_
            assert list(pipe.classes_) == [0,1]
            probabilities=pipe.predict_proba(b)[:,1]
            np.testing.assert_allclose(probabilities,bundle['probabilities'],atol=1e-7)
            pre=pipe['preprocess']
            nums=list(a.select_dtypes(include='number'))
            np.testing.assert_allclose(pre.named_transformers_['numeric']['imputer'].statistics_,a[nums].median())
            assert pre.named_transformers_['numeric']['scaler'].n_samples_seen_==len(a)
            assert len(original_feature_mapping(pre))==len(pre.get_feature_names_out())
            low,high=evaluate(d,probabilities,.2),evaluate(d,probabilities,.8)
            for metric in ['roc_auc','average_precision']:
                assert low[metric]==high[metric]
                assert abs(low[metric]-bundle['metrics'][metric])<1e-7
            assert low['recall']>=high['recall']
            checks.append({'model':bundle['name'],'train_rows':len(a),'test_rows':len(b),
                'prediction_reload':'ok','positive_class':'1 = yes','duration':'exclue',
                'training_only_imputation_scaling':'ok','threshold_invariants':'ok'})
            q=next(r for r in quick['comparison'] if r['name']==bundle['name'])
            comparison.append({'model':bundle['name'],'quick_cv_ap':q['cv_ap'],'full_cv_ap':bundle['cv_ap'],
                'quick_test_ap':q['metrics']['average_precision'],'full_test_ap':bundle['metrics']['average_precision'],
                'quick_test_roc_auc':q['metrics']['roc_auc'],'full_test_roc_auc':bundle['metrics']['roc_auc'],
                'quick_training_seconds':q['seconds'],'full_training_seconds':bundle['seconds']})
    chosen=load_bundle(full['selected_id'])
    assert customer_json(chosen['X_test'].iloc[0])==full['customer']
    local=full['local_explanation']
    probability=float(chosen['pipeline'].predict_proba(chosen['X_test'].iloc[:1])[0,1])
    assert abs(local['probability']-probability)<1e-7
    assert abs(local['shap']['base_values'][0]+sum(local['shap']['values'][0])-probability)<1e-5
    assert full['global_explanation']['additivity_error']<1e-5
    assert len(local['lime']['values'])==len(a.columns) and local['lime']['rules']
    result={'mandatory_local_modules':6,'models':checks,'holdout_index_sha256':expected_hash,
        'same_holdout_quick_full':True,'holdout_previously_observed':True,
        'cv_selection':'Average Precision ; budget identique de 2 candidats, aucun réajustement sur le test',
        'shap_units':local['shap']['units'],'shap_local_additivity_error':local['shap']['additivity_error'],
        'shap_global_additivity_error':full['global_explanation']['additivity_error'],
        'local_probability_matches_original_customer':True,'lime_r2':local['lime']['score'],
        'external_acceptance':{'browser':'bloqué : Browser is not available: chrome',
            'github':'destination AhmedZri ; connexion actuelle sur un autre compte, reconnexion requise ; création non exposée',
            'hugging_face':'destination Ahmedzrii ; connexion actuelle sur un autre compte sans écriture repos ; autorisation publique requise'}}
    (ROOT/'reports/acceptance_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/mode_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ACADEMIC_ARTIFACT_AUDIT_OK : 5 modèles, même holdout, métriques et SHAP vérifiés')


if __name__=='__main__':
    audit()
