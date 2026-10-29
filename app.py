"""Présentation Streamlit ; inférence et interprétation via HTTP FastAPI."""
import json
import os
from uuid import uuid4
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import load_uci, read_csv, prepare_data, split_data
from src.modeling import train_models, MODEL_NAMES
from src.model_registry import save_bundle, load_bundle
from src.evaluation import evaluate, fairness_groups
from src.api_client import api_request, customer_json
from src.plots import shap_figure

st.set_page_config(page_title='BankInsight XAI', page_icon='🏦', layout='wide')
st.markdown('''<style>
.stApp{background:#f4f7fb} h1,h2,h3{color:#142c50}
[data-testid="stSidebar"]{background:#e7edf6}
[data-testid="stMetric"]{background:white;border:1px solid #dce4ee;border-radius:10px;padding:16px}
.stButton>button{border-radius:8px} .stMainBlockContainer{padding-top:2rem}
</style>''', unsafe_allow_html=True)
st.title('BankInsight XAI')
st.caption('Classification explicable pour le marketing bancaire • ENSI • Data Mining 2026/2027')
st.write('Comparer les modèles, comprendre les associations et préparer une campagne de dépôts à terme.')
sections = ['1 · Données', '2 · Préparation', '3 · Modélisation', '4 · Évaluation', '5 · Explainable AI', '6 · Service API']
section = st.sidebar.radio('Parcours de démonstration', sections)
source = st.sidebar.radio('Source des données', ['UCI Bank Marketing', 'Mon CSV'])
try:
    if source == 'UCI Bank Marketing':
        @st.cache_data(show_spinner='Chargement du jeu UCI…')
        def uci():
            return load_uci()
        frame = uci()
        st.sidebar.caption('Moro, Rita et Cortez · UCI 222 · CC BY 4.0')
    else:
        st.sidebar.info('Les CSV restent locaux. Sur un hébergement public partagé, utiliser uniquement des données publiques ou fictives.')
        upload = st.sidebar.file_uploader('Importer un CSV UTF-8', type=['csv'])
        if upload is None:
            st.info('Importer un fichier CSV dans le panneau de gauche pour commencer.')
            st.stop()
        frame = read_csv(upload)
    target_options = ['y'] if source == 'UCI Bank Marketing' else list(frame.columns)
    target = st.sidebar.selectbox('Variable cible', target_options,
        index=target_options.index('y') if 'y' in target_options else len(target_options)-1,
        disabled=source == 'UCI Bank Marketing')
    classes = frame[target].dropna().unique().tolist()
    if len(classes) != 2:
        st.error('La cible doit avoir exactement deux classes. Choisir une autre colonne.')
        st.stop()
    positive_options = ['yes'] if source == 'UCI Bank Marketing' else classes
    positive = st.sidebar.selectbox('Classe positive', positive_options,
        index=positive_options.index('yes') if 'yes' in positive_options else 1,
        disabled=source == 'UCI Bank Marketing')
    X, y = prepare_data(frame, target, positive)
except (ValueError, RuntimeError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.warning('duration exclue : sa valeur est connue après l’appel.')
st.sidebar.caption('Décision : juste avant l’appel planifié. Canal, jour, mois et rang de contact supposés connus. Voir docs/methodologie.md.')
settings = st.session_state.setdefault('settings', dict(test_size=.2, seed=42, train_limit=2500,
    numeric_strategy='median', categorical_strategy='most_frequent', scale=True))

if section == sections[0]:
    a, b, c = st.columns(3)
    a.metric('Observations', f'{len(frame):,}')
    b.metric('Variables autorisées', X.shape[1])
    c.metric('Classe positive', f'{y.mean():.1%}')
    st.info(f'Cible : {target} = {positive}. Source : ' + ('UCI Bank Marketing, DOI 10.24432/C5K306.' if source.startswith('UCI') else 'CSV utilisateur, provenance à documenter.'))
    st.dataframe(frame.head(50), width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(frame, x=target, title='Répartition des classes'))
    r.dataframe(pd.DataFrame({'Type': frame.dtypes.astype(str), 'Manquantes': frame.isna().sum(), '% manquantes': frame.isna().mean()*100}))
    st.dataframe(frame.describe(include='all').T, width="stretch")
    variable = st.selectbox('Variable à explorer', X.columns)
    st.plotly_chart(px.histogram(X, x=variable, nbins=35, title=f'Distribution : {variable}'))
    group = pd.qcut(X[variable], 8, duplicates='drop').astype(str) if pd.api.types.is_numeric_dtype(X[variable]) and X[variable].nunique() > 12 else X[variable].fillna('manquant').astype(str)
    rates = pd.DataFrame({'groupe': group, 'Souscription': y}).groupby('groupe', observed=True).Souscription.agg(['mean', 'count']).reset_index()
    st.plotly_chart(px.bar(rates, x='groupe', y='mean', hover_data=['count'], title='Taux positif par groupe (association descriptive)', labels={'mean': 'Taux positif'}))
    numeric = X.select_dtypes(include='number')
    if len(numeric.columns) > 1:
        st.plotly_chart(px.imshow(numeric.corr(), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Corrélations numériques'))
    st.caption('« unknown » constitue une modalité inconnue conservée ; pdays = -1 signifie aucun contact antérieur, ce n’est pas une valeur manquante.')

elif section == sections[1]:
    st.subheader('Préparer un protocole reproductible')
    features = st.multiselect('Variables explicatives (duration indisponible)', X.columns,
        default=[c for c in st.session_state.get('features', list(X.columns)) if c in X])
    test_size = st.slider('Part du test indépendant', .1, .4, float(settings['test_size']), .05)
    seed = st.number_input('Graine aléatoire', 0, 100000, int(settings['seed']))
    quick = st.radio('Volume d’entraînement', ['Démonstration : 2 500 lignes maximum', 'Toutes les données'], index=0 if settings['train_limit'] else 1)
    num = st.selectbox('Imputation numérique', ['median', 'mean'], index=['median','mean'].index(settings['numeric_strategy']), format_func=lambda v: {'median':'Médiane', 'mean':'Moyenne'}[v])
    cat = st.selectbox('Imputation catégorielle', ['most_frequent', 'constant'], index=['most_frequent','constant'].index(settings['categorical_strategy']), format_func=lambda v: {'most_frequent':'Modalité majoritaire', 'constant':'Modalité « manquant »'}[v])
    scale = st.checkbox('Standardiser les variables numériques', value=settings['scale'])
    st.info('Encodage one-hot systématique ; catégories inconnues ignorées. Les imputations et la standardisation sont apprises uniquement sur chaque partition d’entraînement. Le test complet est séparé avant tout échantillonnage.')
    if st.button('Appliquer la préparation', type='primary'):
        try:
            xx, yy = prepare_data(frame, target, positive, features)
            parts = split_data(xx, yy, test_size, int(seed), 2500 if quick.startswith('Démonstration') else None)
            st.session_state['features'] = features
            st.session_state['settings'] = dict(test_size=test_size, seed=int(seed), train_limit=2500 if quick.startswith('Démonstration') else None,
                numeric_strategy=num, categorical_strategy=cat, scale=scale)
            st.success(f'Configuration enregistrée : {len(parts[0])} lignes d’entraînement, {len(parts[1])} lignes de test.')
        except ValueError as exc:
            st.error(str(exc))

else:
    try:
        metadata = api_request('/api/models')
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    # Never compare archived models from different preparation protocols.
    if metadata and section != sections[5]:
        active_ids = st.session_state.get('experiment_ids')
        if active_ids:
            metadata = [m for m in metadata if m['model_id'] in active_ids]
        if not active_ids:
            protocols = list(dict.fromkeys(json.dumps(m['config'], sort_keys=True) for m in metadata))
            protocol = st.selectbox('Expérience archivée', protocols, format_func=lambda text:
                f"{json.loads(text).get('source', 'Données')} · train {json.loads(text).get('train_rows', '?')} · graine {json.loads(text).get('seed', '?')} · budget {json.loads(text).get('search_budget', '?')} · {json.loads(text).get('experiment_id', 'démo')[:8]}")
            metadata = [m for m in metadata if json.dumps(m['config'], sort_keys=True) == protocol]
        elif st.button('Parcourir les autres expériences archivées'):
            del st.session_state['experiment_ids']
            st.rerun()
    if section == sections[2]:
        st.subheader('Comparer cinq familles de modèles')
        st.json(settings, expanded=False)
        names = st.multiselect('Modèles à entraîner', list(MODEL_NAMES), default=list(MODEL_NAMES), format_func=MODEL_NAMES.get)
        budget = st.slider('Candidats par recherche (3 plis stratifiés)', 1, 9, 2)
        overrides = {}
        with st.expander('Hyperparamètres avancés : fixer une valeur hors recherche'):
            manual = st.checkbox('Activer les valeurs fixes')
            if manual:
                overrides = {'logistic': {'C': st.number_input('C logistique', .01, 100., 1.)},
                    'tree': {'max_depth': st.slider('Profondeur arbre', 2, 20, 6)},
                    'knn': {'n_neighbors': st.slider('Voisins KNN', 3, 31, 15, 2)},
                    'forest': {'n_estimators': st.slider('Arbres forêt', 50, 300, 100, 50)},
                    'xgboost': {'n_estimators': st.slider('Arbres XGBoost', 50, 300, 100, 50)}}
        st.write('Sélection par Average Precision moyenne en validation croisée sur l’entraînement. Pondération équilibrée pour logistique, arbre et forêt ; poids positif pour XGBoost. KNN utilise les distances sans pondération de classe, d’où l’importance de PR-AUC et du seuil.')
        if st.button('Entraîner et comparer', type='primary', disabled=not names):
            features = st.session_state.get('features', list(X.columns))
            features = [c for c in features if c in X]
            try:
                xx, yy = prepare_data(frame, target, positive, features)
                train, test, yt, yv = split_data(xx, yy, settings['test_size'], settings['seed'], settings['train_limit'])
                bar = st.progress(0, text='Validation croisée en cours…')
                done = []
                def progress(name):
                    done.append(name)
                    bar.progress(len(done)/len(names), text=f'{MODEL_NAMES[name]} terminé')
                with st.spinner('Entraînement et recherche des hyperparamètres…'):
                    results = train_models(train, test, yt, yv, names, seed=settings['seed'], search_budget=budget,
                        numeric_strategy=settings['numeric_strategy'], categorical_strategy=settings['categorical_strategy'],
                        scale=settings['scale'], overrides=overrides, progress=progress)
                config = dict(settings, source=source, target=target, positive=str(positive), search_budget=budget,
                              train_rows=len(train), test_rows=len(test), overrides=overrides, experiment_id=uuid4().hex)
                ids = [save_bundle(result, train, test, yv, config) for result in results]
                st.session_state['experiment_ids'] = ids
                st.success('Modèles enregistrés. Ouvrir Évaluation ou Explainable AI.')
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        if metadata:
            current_ids = st.session_state.get('experiment_ids')
            current = [m for m in metadata if not current_ids or m['model_id'] in current_ids]
            table = pd.DataFrame([{'Modèle': MODEL_NAMES[m['name']], 'AP validation': m['cv_ap'], 'Écart-type CV': m['cv_std'],
                'AP test': m['metrics']['average_precision'], 'ROC-AUC test': m['metrics']['roc_auc'], 'Durée (s)': m['seconds']} for m in current])
            st.dataframe(table.sort_values('AP validation', ascending=False), width="stretch")
            st.caption('Seule la colonne AP validation détermine le modèle recommandé. Les résultats archivés peuvent appartenir à plusieurs expériences ; lancer un nouvel entraînement pour isoler une comparaison.')
    elif section == sections[5]:
        st.subheader('Service FastAPI utilisé par cette application')
        st.json(api_request('/api/health'))
        if os.environ.get('BANKINSIGHT_API_INTERNAL') == '1':
            specification = api_request('/api/openapi.json')
            st.download_button('Télécharger la spécification OpenAPI',
                json.dumps(specification, ensure_ascii=False, indent=2),
                file_name='bankinsight_openapi.json', mime='application/json')
            st.caption('Sur Community Cloud, FastAPI est interne au serveur. La spécification provient de son endpoint HTTP ; son adresse localhost n’est pas accessible depuis votre navigateur.')
            with st.expander('Consulter les endpoints OpenAPI'):
                st.json(specification, expanded=False)
        else:
            st.link_button('Documentation OpenAPI', os.environ.get('BANKINSIGHT_PUBLIC_API_DOCS', 'http://127.0.0.1:8000/api/docs'))
        st.write('La liste ci-dessous et les prédictions/explications de la section XAI proviennent de requêtes HTTP réelles.')
        st.json(metadata, expanded=False)
        st.code('GET /api/health\nGET /api/models\nPOST /api/predict\nPOST /api/explain/local\nPOST /api/explain/global')
        if metadata:
            bundle = load_bundle(metadata[0]['model_id'])
            st.code(json.dumps({'model_id': bundle['model_id'], 'customers': [customer_json(bundle['X_test'].iloc[0])], 'threshold': .5}, ensure_ascii=False, indent=2), language='json')
    else:
        if not metadata:
            st.info('Entraîner au moins un modèle dans la section Modélisation.')
            st.stop()
        ids = st.session_state.get('experiment_ids')
        current = [m for m in metadata if not ids or m['model_id'] in ids]
        best = max(current, key=lambda m: m['cv_ap'])
        st.success(f"Recommandation CV : {MODEL_NAMES[best['name']]} (AP = {best['cv_ap']:.3f}).")
        selected = st.selectbox('Modèle à examiner', current, format_func=lambda m: MODEL_NAMES[m['name']] + ' · ' + m['model_id'][:8], index=current.index(best))
        bundle = load_bundle(selected['model_id'])
        st.caption(f"Expérience : {bundle['config'].get('source', 'UCI')} · entraînement {len(bundle['X_train'])} · test {len(bundle['X_test'])} · cible positive {bundle['config'].get('positive', 'yes')}")
        threshold = st.slider('Seuil de décision exploratoire', .01, .99, .5, .01)
        st.warning('Modifier le seuil sur le test est exploratoire : ce réglage ne constitue pas une optimisation validée. AP et ROC-AUC ne dépendent pas du seuil.')
        if section == sections[3]:
            m = evaluate(bundle['y_test'], bundle['probabilities'], threshold)
            cols = st.columns(5)
            for col, label, key in zip(cols, ['Average Precision', 'ROC-AUC', 'Précision', 'Rappel', 'F1'], ['average_precision', 'roc_auc', 'precision', 'recall', 'f1']):
                col.metric(label, f'{m[key]:.3f}')
            st.info('Average Precision résume la précision obtenue aux niveaux de rappel successifs (référence aléatoire : taux positif). ROC-AUC mesure le classement positif/négatif ; elle peut paraître élevée malgré de nombreux faux positifs. AP est notre mesure de PR-AUC, différente de l’aire trapézoïdale.')
            roc, pr = go.Figure(), go.Figure()
            rows = []
            for meta in current:
                b = load_bundle(meta['model_id'])
                scores = evaluate(b['y_test'], b['probabilities'], threshold)
                label = MODEL_NAMES[meta['name']]
                roc.add_trace(go.Scatter(x=scores['roc']['fpr'], y=scores['roc']['tpr'], name=label))
                pr.add_trace(go.Scatter(x=scores['pr']['recall'], y=scores['pr']['precision'], name=label))
                rows.append(dict(Modèle=label, **{k: scores[k] for k in ['average_precision','roc_auc','precision','recall','f1']}))
            roc.update_layout(title='Courbes ROC superposées', xaxis_title='Taux de faux positifs', yaxis_title='Rappel')
            pr.update_layout(title='Courbes précision-rappel', xaxis_title='Rappel', yaxis_title='Précision')
            l, r = st.columns(2)
            l.plotly_chart(roc)
            r.plotly_chart(pr)
            st.dataframe(pd.DataFrame(rows), width="stretch")
            st.plotly_chart(px.imshow(m['confusion'], text_auto=True, x=['Prédit négatif','Prédit positif'], y=['Réel négatif','Réel positif'], color_continuous_scale='Blues', title='Matrice de confusion'))
            st.dataframe(pd.DataFrame(m['report']).T)
            st.write('Faux positif : sollicitation sans souscription, donc coût commercial. Faux négatif : client susceptible de souscrire qui ne serait pas ciblé. Le coût de chaque erreur doit guider une future validation du seuil sur l’entraînement.')
        else:
            st.subheader('Comprendre le modèle et un client')
            sample_size = st.slider('Nombre de clients pour SHAP global', 8, 48, 24, 8)
            st.caption('SHAP permutation marginale : 20 références d’entraînement, 3 permutations par client ; corrélations non conditionnées. LIME : 700 perturbations et catégories discrètes. Les deux expliquent la même probabilité positive. Les petits échantillons limitent la stabilité des classements.')
            if st.button('Calculer les explications globales'):
                try:
                    with st.spinner('Appel FastAPI : SHAP et permutation…'):
                        st.session_state['global_' + selected['model_id']] = api_request('/api/explain/global', {'model_id': selected['model_id'], 'sample_size': sample_size})
                except RuntimeError as exc:
                    st.error(str(exc))
            global_result = st.session_state.get('global_' + selected['model_id'])
            if global_result:
                g = global_result
                st.pyplot(shap_figure(g))
                feature = st.selectbox('Variable de dépendance SHAP', [r[0] for r in g['ranking']])
                st.pyplot(shap_figure(g, 'dependence', feature))
                st.json(g['category_labels'], expanded=False)
                st.caption('Codes catégoriels arbitraires : leur couleur et leur ordre numérique ne représentent pas un ordre métier.')
                st.dataframe(pd.DataFrame({'Variable métier': g['features'], 'Moyenne |SHAP|': np.abs(g['values']).mean(axis=0), 'Permutation ΔAP': g['permutation'], 'Écart-type permutation': g['permutation_std']}))
                st.caption(f"Permutation : {g['permutation_rows']} observations, 3 répétitions ; évaluation descriptive du test, sans sélection de variables sur ce résultat.")
                if g['native']:
                    st.bar_chart(pd.Series(g['native'], name='Importance native agrégée'))
                    st.caption('Somme des valeurs absolues par variable métier ; ' + g['native_units'])
                    with st.expander('Coefficients signés / importances encodées'):
                        st.dataframe(pd.DataFrame(g['native_detail']))
                else:
                    st.info('KNN ne possède pas d’importance native. Utiliser SHAP et permutation.')
            mode = st.radio('Client à expliquer', ['Client du test', 'Client fictif'], horizontal=True)
            index = st.number_input('Position dans le test', 0, len(bundle['X_test'])-1, 0)
            row = bundle['X_test'].iloc[int(index)].copy()
            if mode == 'Client fictif':
                with st.form('customer'):
                    for c in row.index:
                        if pd.api.types.is_numeric_dtype(bundle['X_train'][c]):
                            row[c] = st.number_input(c, value=float(row[c]) if pd.notna(row[c]) else 0.)
                        else:
                            choices = sorted(bundle['X_train'][c].dropna().astype(str).unique().tolist())
                            row[c] = st.selectbox(c, choices, index=choices.index(str(row[c])) if str(row[c]) in choices else 0)
                    submitted = st.form_submit_button('Prédire et expliquer ce client')
            else:
                submitted = st.button('Prédire et expliquer ce client')
            customer = customer_json(row)
            st.dataframe(pd.DataFrame([customer]))
            if submitted:
                try:
                    with st.spinner('Appels FastAPI : prédiction, SHAP et LIME…'):
                        prediction = api_request('/api/predict', {'model_id': selected['model_id'], 'customers': [customer], 'threshold': threshold})
                        local = api_request('/api/explain/local', {'model_id': selected['model_id'], 'customer': customer, 'threshold': threshold})
                    st.session_state['local_result'] = dict(model_id=selected['model_id'], customer=customer, prediction=prediction, explanation=local)
                except RuntimeError as exc:
                    st.error(str(exc))
            saved = st.session_state.get('local_result')
            if saved and saved['model_id'] == selected['model_id'] and saved['customer'] == customer:
                local = saved['explanation']
                probability = saved['prediction']['probabilities'][0]
                st.metric('Probabilité positive (API)', f'{probability:.1%}')
                st.write('Décision au seuil courant : ' + ('cibler' if probability >= threshold else 'ne pas cibler'))
                st.pyplot(shap_figure(local['shap'], 'waterfall'))
                st.dataframe(pd.DataFrame(local['lime']['rules'], columns=['Règle locale LIME', 'Contribution au substitut']))
                st.write(local['summary'])
                st.write(f"Fidélité locale LIME R² : {local['lime']['score']:.3f}. Accord de signe sur les principales variables SHAP : {local['top_sign_agreement']}. Erreur d’additivité SHAP : {local['shap']['additivity_error']:.2g}.")
                if local['lime']['score'] < .5:
                    st.warning('Fidélité locale LIME faible : interpréter ses poids avec prudence, même si leurs signes concordent avec SHAP.')
                st.caption('Les poids LIME appartiennent à un substitut local régularisé ; ils ne sont pas une décomposition additive exacte de la probabilité. SHAP et LIME peuvent diverger selon le voisinage, les corrélations et les approximations.')
            with st.expander('Analyse critique, équité et recommandations', expanded=True):
                sensitive = [c for c in ['age', 'marital'] if c in bundle['X_test']]
                if sensitive:
                    group = st.selectbox('Audit exploratoire des groupes', sensitive)
                    st.dataframe(pd.DataFrame(fairness_groups(bundle['X_test'], bundle['y_test'], bundle['probabilities'], group, threshold)))
                st.write('Âge et situation familiale peuvent influencer le ciblage directement ou via des variables corrélées. Les petits groupes et les historiques de sollicitation rendent ces comparaisons incertaines ; elles ne certifient pas l’équité et ne justifient pas une exclusion de clients.')
                st.write('La logistique facilite l’examen des coefficients ; XGBoost peut représenter des interactions mais nécessite davantage de contrôles. Comparer leurs AP en CV et leur stabilité avant de choisir. Les résultats du test ne prouvent pas un gain commercial futur.')
                st.write('Recommandations : tester la campagne sur un échantillon prospectif, fixer les coûts de sollicitation et d’opportunité, auditer les groupes sensibles et conserver une revue humaine. Une association SHAP ne démontre pas qu’une modification de la variable ferait souscrire le client. Les probabilités pondérées ne sont pas calibrées ; une calibration sur l’entraînement serait nécessaire pour un budget fondé sur des probabilités absolues.')
