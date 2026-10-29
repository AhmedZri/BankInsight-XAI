"""Démarrage supervisé local, Windows et Linux."""
from pathlib import Path
import os
import argparse
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-port', type=int, default=8000)
    parser.add_argument('--ui-port', type=int, default=8501)
    args = parser.parse_args()
    for port in [args.api_port, args.ui_port]:
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'Port {port} occupé ; choisir --api-port et --ui-port libres.') from exc
    os.environ['BANKINSIGHT_API_URL'] = f'http://127.0.0.1:{args.api_port}'
    os.environ['BANKINSIGHT_PUBLIC_API_DOCS'] = f'http://127.0.0.1:{args.api_port}/api/docs'
    children = []
    def stop():
        for child in reversed(children):
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
    def signal_stop(*_):
        stop()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, signal_stop)
    signal.signal(signal.SIGTERM, signal_stop)
    try:
        children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', str(args.api_port)], cwd=ROOT))
        for _ in range(90):
            if children[0].poll() is not None:
                raise RuntimeError('FastAPI a quitté prématurément. Vérifier le port 8000.')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{args.api_port}/api/health', timeout=1)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('FastAPI ne répond pas après 90 secondes.')
        children.append(subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', str(args.ui_port), '--server.headless', 'true'], cwd=ROOT, env=os.environ.copy()))
        print(f'Application : http://127.0.0.1:{args.ui_port} | API : http://127.0.0.1:{args.api_port}/api/docs', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
    finally:
        stop()


if __name__ == '__main__':
    main()
