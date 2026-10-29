"""Cycle de vie du service HTTP interne, sans logique ML supplémentaire."""
import atexit
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def startup_lock(path, timeout):
    """Verrou noyau libéré même si un processus de démarrage s'arrête brutalement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        deadline = time.monotonic() + timeout
        while True:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Un autre processus initialise encore l’API. Réessayer.')
                time.sleep(.1)
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


class ApiService:
    """Une API par port/projet, réutilisée à chaque rerun et redémarrée si elle meurt."""
    def __init__(self, port=8765, runtime_dir=None, timeout=60):
        if not 1024 <= port <= 65535:
            raise ValueError('Le port interne doit être compris entre 1024 et 65535.')
        self.port, self.timeout = port, timeout
        self.url = f'http://127.0.0.1:{port}'
        self.runtime_dir = Path(runtime_dir or os.environ.get('BANKINSIGHT_CLOUD_RUNTIME_DIR', ROOT/'tmp/cloud_runtime'))
        registry = str(Path(os.environ.get('BANKINSIGHT_REGISTRY', ROOT/'artifacts/registry')).resolve())
        self.identity = hashlib.sha256((str(ROOT) + registry).encode()).hexdigest()
        self._process = None
        self._mutex = threading.RLock()
        atexit.register(self.stop)

    def healthy(self):
        try:
            with build_opener(ProxyHandler({})).open(self.url+'/api/health', timeout=2) as response:
                return response.status == 200 and response.headers.get('X-BankInsight-Instance') == self.identity
        except (OSError, URLError):
            return False

    def ensure_running(self):
        with self._mutex, startup_lock(self.runtime_dir/f'api-{self.port}.lock', self.timeout):
            if self.healthy():
                return self.url
            if self._process is None or self._process.poll() is not None:
                try:
                    with socket.create_connection(('127.0.0.1', self.port), timeout=.3):
                        raise RuntimeError(f'Le port interne {self.port} est occupé par un autre service.')
                except (ConnectionRefusedError, TimeoutError):
                    pass
                env = os.environ.copy()
                env['BANKINSIGHT_CLOUD_INSTANCE'] = self.identity
                self._process = subprocess.Popen(
                    [sys.executable, '-u', '-m', 'src.cloud_runtime', '--serve', '--port', str(self.port)],
                    cwd=ROOT, env=env, stdin=subprocess.PIPE)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.healthy():
                    return self.url
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f'L’API a quitté au démarrage (code {self._process.returncode}). Voir les journaux.')
                time.sleep(.1)
            raise RuntimeError('L’API ne répond pas dans le délai de démarrage. Voir les journaux.')

    def stop(self):
        # Un processus qui réutilise l’API d’un autre propriétaire ne l’arrête jamais.
        with self._mutex:
            if self._process is not None:
                if self._process.stdin is not None and not self._process.stdin.closed:
                    self._process.stdin.close()
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()


def serve(port):
    import asyncio
    import uvicorn
    from src.api.main import app

    gate = asyncio.Semaphore(1)
    identity = os.environ['BANKINSIGHT_CLOUD_INSTANCE']

    @app.middleware('http')
    async def cloud_requests(request, call_next):
        # Un seul chargement/calcul ML simultané ; santé et documentation restent disponibles.
        if request.method == 'POST':
            async with gate:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers['X-BankInsight-Instance'] = identity
        response.headers['X-BankInsight-Pid'] = str(os.getpid())
        return response

    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, workers=1, access_log=False))

    def parent_lease():
        # EOF à l’arrêt normal ou brutal du parent : pas de processus API orphelin.
        sys.stdin.buffer.read()
        server.should_exit = True

    threading.Thread(target=parent_lease, daemon=True).start()
    server.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--port', type=int, required=True)
    serve(parser.parse_args().port)
