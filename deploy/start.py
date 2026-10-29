"""Un seul conteneur, trois processus supervisés et arrêt coordonné."""
import os
import signal
import subprocess
import sys
import time

os.environ['BANKINSIGHT_PUBLIC_API_DOCS'] = '/api/docs'
commands = [[sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', '8000'],
    [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true', '--server.fileWatcherType', 'none'],
    ['nginx', '-c', '/app/deploy/nginx.conf', '-g', 'daemon off;']]
children = []
exit_code = 0


def stop(*_):
    for child in reversed(children):
        child.terminate()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    for command in commands:
        children.append(subprocess.Popen(command))
    while all(c.poll() is None for c in children):
        time.sleep(1)
    exit_code = next((c.returncode for c in children if c.poll() is not None), 1)
finally:
    stop()
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
sys.exit(exit_code)
