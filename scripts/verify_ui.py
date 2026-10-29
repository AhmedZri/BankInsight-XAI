"""Parcours Streamlit sans navigateur ; nécessite l'API réellement démarrée."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src.model_registry import list_models

root = Path(__file__).resolve().parents[1]
at = AppTest.from_file(str(root/'app.py'), default_timeout=180).run()
assert not at.exception, [e.message for e in at.exception]
print('Données OK', flush=True)
at.sidebar.radio[0].set_value('2 · Préparation').run()
next(b for b in at.button if b.label=='Appliquer la préparation').click().run()
assert not at.exception and not at.error
assert at.session_state['settings']['test_size']==.2
at.sidebar.radio[0].set_value('3 · Modélisation').run()
next(b for b in at.button if b.label=='Entraîner et comparer').click().run()
assert not at.exception, [e.message for e in at.exception]
assert not at.error, [e.value for e in at.error]
assert len(at.session_state['experiment_ids'])==5
print('Préparation appliquée et cinq modèles entraînés depuis l’interface OK',flush=True)
for value in ['2 · Préparation', '3 · Modélisation', '4 · Évaluation', '5 · Explainable AI', '6 · Service API']:
    at.sidebar.radio[0].set_value(value).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    print(value, 'OK', flush=True)
at.sidebar.radio[0].set_value('4 · Évaluation').run()
scores=[m.value for m in at.metric[:2]]
next(s for s in at.slider if s.label=='Seuil de décision exploratoire').set_value(.2).run()
assert scores==[m.value for m in at.metric[:2]]
next(s for s in at.slider if s.label=='Seuil de décision exploratoire').set_value(.8).run()
assert scores==[m.value for m in at.metric[:2]]
print('Seuil interactif : AP et ROC-AUC invariantes OK',flush=True)
next(b for b in at.button if b.label=='Parcourir les autres expériences archivées').click().run()
full=next(m for m in list_models() if m['config'].get('train_rows')==36168)
next(s for s in at.selectbox if s.label=='Expérience archivée').set_value(json.dumps(full['config'],sort_keys=True)).run()
at.sidebar.radio[0].set_value('5 · Explainable AI').run()
button = next(b for b in at.button if b.label == 'Calculer les explications globales')
button.click().run()
assert not at.exception and not at.error
assert any(key.startswith('global_') for key in at.session_state.filtered_state)
print('Affichage SHAP global et dépendance OK', flush=True)
button = next(b for b in at.button if b.label == 'Prédire et expliquer ce client')
button.click().run()
assert not at.exception, [e.message for e in at.exception]
assert not at.error, [e.value for e in at.error]
assert at.session_state['local_result']['explanation']['lime']['rules']
print('Inférence HTTP, waterfall et LIME OK', flush=True)
next(r for r in at.radio if r.label=='Client à expliquer').set_value('Client fictif').run()
next(b for b in at.button if b.label=='Prédire et expliquer ce client').click().run()
assert not at.exception, [e.message for e in at.exception]
assert not at.error, [e.value for e in at.error]
print('Formulaire fictif et explication via API OK',flush=True)
print('STREAMLIT_APPTEST_OK', flush=True)
