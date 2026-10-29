"""Recette de cloud_app.py sur modèles archivés, sans réentraînement UCI."""
from pathlib import Path
import hashlib
import json
import os
import socket
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psutil
import requests
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def frozen_hashes():
    files=[p for folder in ['artifacts','data','reports','notebooks']
           for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'src'/name for name in ['data.py','preprocessing.py','modeling.py',
        'evaluation.py','explainability.py','model_registry.py','plots.py']]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate():
    baseline=frozen_hashes()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    os.environ['BANKINSIGHT_CLOUD_API_PORT']=str(port)
    peak=[0]
    finished=threading.Event()
    def memory_monitor():
        process=psutil.Process()
        while not finished.is_set():
            try:
                peak[0]=max(peak[0],sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running()))
            except psutil.NoSuchProcess:
                pass
            finished.wait(.1)
    threading.Thread(target=memory_monitor,daemon=True).start()
    started=time.monotonic()
    try:
        at=AppTest.from_file(str(ROOT/'cloud_app.py'),default_timeout=240).run()
        assert not at.exception, [e.message for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        client=requests.Session()
        client.trust_env=False
        base=f'http://127.0.0.1:{port}'
        health=client.get(base+'/api/health',timeout=5)
        assert health.status_code==200 and health.json()['status']=='ok'
        pid=health.headers['X-BankInsight-Pid']
        evidence=json.loads((ROOT/'reports/experiment.json').read_text(encoding='utf-8'))
        payload={'model_id':evidence['selected_id'],'customers':[evidence['customer']],'threshold':.5}
        predicted=client.post(base+'/api/predict',json=payload,timeout=60)
        assert predicted.status_code==200,predicted.text
        assert abs(predicted.json()['probabilities'][0]-evidence['local_explanation']['probability'])<1e-7
        for section in ['2 · Préparation','3 · Modélisation','4 · Évaluation','5 · Explainable AI','6 · Service API']:
            at.sidebar.radio[0].set_value(section).run()
            assert not at.exception, [e.message for e in at.exception]
            assert not at.error, [e.value for e in at.error]
            assert client.get(base+'/api/health',timeout=5).headers['X-BankInsight-Pid']==pid
            print(section+' OK',flush=True)
        assert at.get('download_button') and not at.get('link_button')
        specification=client.get(base+'/api/openapi.json',timeout=10)
        assert specification.status_code==200 and '/api/explain/local' in specification.json()['paths']
        assert client.get(base+'/api/docs',timeout=10).status_code==200
        at.sidebar.radio[0].set_value('5 · Explainable AI').run()
        next(b for b in at.button if b.label=='Calculer les explications globales').click().run()
        assert not at.exception and not at.error
        assert at.session_state['global_'+evidence['selected_id']]['additivity_error']<1e-5
        next(b for b in at.button if b.label=='Prédire et expliquer ce client').click().run()
        assert not at.exception, [e.message for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        local=at.session_state['local_result']
        assert local['model_id']==evidence['selected_id']
        assert abs(local['explanation']['probability']-evidence['local_explanation']['probability'])<1e-7
        assert local['explanation']['lime']['rules']
        # Une deuxième session utilise le même service ; aucun chargement de modèle au démarrage du dashboard.
        second=AppTest.from_file(str(ROOT/'cloud_app.py'),default_timeout=240).run()
        assert not second.exception and not second.error
        assert client.get(base+'/api/health',timeout=5).headers['X-BankInsight-Pid']==pid
        assert frozen_hashes()==baseline,'Des fichiers scientifiques ont changé'
        cgroup=Path('/sys/fs/cgroup')
        cgroup_peak=int((cgroup/'memory.peak').read_text()) if (cgroup/'memory.peak').exists() else None
        cgroup_limit=(cgroup/'memory.max').read_text().strip() if (cgroup/'memory.max').exists() else None
        print(json.dumps({'cloud_entrypoint':'cloud_app.py','six_sections':'ok','real_http_prediction':'ok',
            'archived_xgboost_shap_lime':'ok','openapi_download':'ok','shared_api_pid':pid,
            'second_session_same_api':True,'scientific_files_unchanged':len(baseline),
            'cgroup_memory_peak_bytes':cgroup_peak,'cgroup_memory_limit_bytes':cgroup_limit,
            'peak_process_tree_rss_mib':round(peak[0]/1024**2,2),'seconds':round(time.monotonic()-started,2)}),flush=True)
    finally:
        finished.set()


if __name__=='__main__':
    validate()
