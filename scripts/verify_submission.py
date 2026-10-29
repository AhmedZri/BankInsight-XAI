"""Vérifier l'archive finale depuis une extraction indépendante."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--archive',default=str(root/'Challenge1_Classification_Explicable_ENSI.zip'))
archive_path=Path(parser.parse_args().archive).resolve()
with tempfile.TemporaryDirectory(prefix='bankinsight_submission_') as directory:
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        archive.extractall(directory)
    extracted = Path(directory) / 'BankInsight-XAI'
    for command in [[sys.executable, '-m', 'pytest', '-q'],
                    [sys.executable, '-c', "from src.data import load_uci; from src.model_registry import list_models,load_bundle; df=load_uci(); m=list_models(); assert len(df)==45211 and len(m)==5; b=load_bundle(m[0]['model_id']); assert 'duration' not in b['X_train']; print('EXTRACTED_UCI_REGISTRY_OK')"],
                    [sys.executable,'scripts/audit_release.py']]:
        subprocess.run(command, cwd=extracted, check=True)
    print(json.dumps({'archive': str(archive_path), 'crc': 'ok', 'extracted_tests': 'passed', 'uci_cache': 'ok', 'registry': 'ok'}))
