"""Point d'entrée Community Cloud : même interface, vraie API HTTP interne."""
import os
from pathlib import Path
import runpy

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(variable, '2')
os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY', ''), '127.0.0.1', 'localhost']))

import streamlit as st
from src.cloud_runtime import ApiService


@st.cache_resource(show_spinner=False)
def cloud_api(port):
    # Le cache est partagé entre sessions ; le verrou OS protège aussi plusieurs processus.
    return ApiService(port=port)


try:
    service = cloud_api(int(os.environ.get('BANKINSIGHT_CLOUD_API_PORT', '8765')))
    os.environ['BANKINSIGHT_API_URL'] = service.ensure_running()
except (RuntimeError, ValueError, OSError) as exc:
    st.error(f"Démarrage de l’API interne impossible : {exc}. Relancer la page et consulter les journaux Community Cloud.")
    st.stop()

os.environ['BANKINSIGHT_API_INTERNAL'] = '1'
runpy.run_path(str(Path(__file__).resolve().with_name('app.py')), run_name='__main__')
