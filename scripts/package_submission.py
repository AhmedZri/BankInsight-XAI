"""Archive déterministe à liste blanche, sans CSV privés ni modèles non approuvés."""
from pathlib import Path
import json
import argparse
import sys
import tempfile
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT

NAME = 'Challenge1_Classification_Explicable_ENSI.zip'
ESSENTIAL = ['app.py', 'src/api/main.py', 'requirements.txt', 'README.md', 'Dockerfile',
    'notebooks/01_bank_marketing_analysis.ipynb', 'reports/one_pager.pdf', 'tests/test_data_ml.py']


def package(output=None, public=False):
    output = Path(output or ROOT / ('BankInsight-XAI_publication.zip' if public else NAME))
    files = []
    for directory in ['src','tests','scripts','docs','notebooks','reports','deploy','assets','.streamlit','.github']:
        path = ROOT / directory
        if path.exists():
            files.extend(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.log'])
    for name in ['app.py','requirements.txt','requirements-lock.txt','README.md','Dockerfile','.dockerignore','.gitignore','.gitattributes','.env.example','pytest.ini','LICENSE']:
        if (ROOT / name).exists(): files.append(ROOT / name)
    for name in ['data/uci_bank.csv','data/uci_bank.json','artifacts/public_demo.json']:
        files.append(ROOT / name)
    approved = json.loads((ROOT / 'artifacts/public_demo.json').read_text(encoding='utf-8'))
    for model_id in approved:
        for suffix in ['.json','.joblib']:
            path = ROOT / 'artifacts/registry' / (model_id + suffix)
            if not path.exists(): raise FileNotFoundError(path)
            files.append(path)
    for p in files:
        if not p.is_file(): raise FileNotFoundError(p)
    if public:
        files = [p for p in files if not (
            p.name.startswith('Projet_Challenge_1_Classification_Explicable') and p.suffix.lower() == '.pdf')]
    prefix = 'BankInsight-XAI/'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(files)):
            info = zipfile.ZipInfo(prefix + path.relative_to(ROOT).as_posix(), date_time=(2026,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip(): raise RuntimeError('CRC invalide')
        names = archive.namelist()
        assert all(prefix + p in names for p in ESSENTIAL)
        assert not any('/.venv/' in p or '/.git/' in p or '__pycache__' in p for p in names)
        with tempfile.TemporaryDirectory(prefix='bankinsight_extract_') as temp:
            archive.extractall(temp)
            assert all((Path(temp) / prefix / p).is_file() for p in ESSENTIAL)
    print(f'{output} | {output.stat().st_size:,} octets | {len(names)} fichiers | CRC et extraction OK')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', action='store_true', help='Exclure le PDF pédagogique privé de la publication')
    package(public=parser.parse_args().public)
