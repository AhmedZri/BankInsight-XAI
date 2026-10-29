from pathlib import Path
import sys
import os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for name, directory in [('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'jupyter'), ('JUPYTER_CONFIG_DIR','jupyter_config')]:
    path = root / 'tmp' / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(path))
path = root / 'notebooks/01_bank_marketing_analysis.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, resources={'metadata': {'path': str(root)}}, kernel_name='python3')
# Select this interpreter directly, without requiring an undocumented installed kernelspec.
client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client.execute()
nbformat.write(nb, path)
print('NOTEBOOK_EXECUTED_OK')
