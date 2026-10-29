import json
import zipfile
from pypdf import PdfReader
from scripts.generate_report import generate_report
from scripts.package_submission import package, ESSENTIAL
from src.data import ROOT


def test_real_report_one_page(tmp_path):
    path = generate_report(output=tmp_path / 'report.pdf')
    reader = PdfReader(path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert 'BankInsight' in text and 'SHAP' in text and 'LIME' in text
    evidence = json.loads((ROOT/'reports/experiment.json').read_text(encoding='utf-8'))
    selected = next(r for r in evidence['comparison'] if r['name'] == evidence['selected_name'])
    assert f"{selected['metrics']['average_precision']:.3f}" in text


def test_zip_integrity_and_extraction(tmp_path):
    path = package(tmp_path / 'submission.zip')
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all('BankInsight-XAI/' + p in names for p in ESSENTIAL)
        assert not any('/.env' == name[-5:] or '/tmp/' in name or '/.git/' in name for name in names)
        approved = json.loads((ROOT/'artifacts/public_demo.json').read_text(encoding='utf-8'))
        assert all(name.split('/')[-1].removesuffix('.joblib') in approved for name in names if name.endswith('.joblib'))
        archive.extractall(tmp_path/'extracted')
    assert (tmp_path/'extracted/BankInsight-XAI/src/api/main.py').is_file()


def test_public_archive_excludes_professor_pdf(tmp_path):
    path=package(tmp_path/'publication.zip',public=True)
    with zipfile.ZipFile(path) as archive:
        assert not any('Projet_Challenge_1_Classification_Explicable' in name for name in archive.namelist())
        assert 'BankInsight-XAI/reports/one_pager.pdf' in archive.namelist()
        attributes=archive.read('BankInsight-XAI/.gitattributes').decode('utf-8')
        assert '*.csv binary' in attributes and '*.pdf binary' in attributes
