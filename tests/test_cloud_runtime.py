"""Intégration du démarrage partagé avec de vrais processus et requêtes HTTP."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler
from src.cloud_runtime import ApiService

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def server_pid(url):
    with build_opener(ProxyHandler({})).open(url+'/api/health') as response:
        assert json.load(response)['status']=='ok'
        return int(response.headers['X-BankInsight-Pid'])


def test_shared_service_reuse_restart_and_cleanup(tmp_path):
    service=ApiService(port=free_port(),runtime_dir=tmp_path)
    try:
        url=service.ensure_running()
        pid=server_pid(url)
        for _ in range(3):
            assert service.ensure_running()==url and server_pid(url)==pid
        borrowed=ApiService(port=service.port,runtime_dir=tmp_path)
        assert borrowed.ensure_running()==url
        borrowed.stop()
        assert service.healthy()  # Un client ne tue pas le service du propriétaire.
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+15
        while service._process.poll() is None and time.monotonic()<deadline:
            time.sleep(.05)
        assert service._process.poll() is not None
        assert service.ensure_running()==url
        assert server_pid(url)!=pid
    finally:
        service.stop()
    deadline=time.monotonic()+10
    while service.healthy() and time.monotonic()<deadline:
        time.sleep(.05)
    assert not service.healthy()


def test_process_safe_simultaneous_startup(tmp_path):
    port=free_port()
    program='''
import sys,json
from src.cloud_runtime import ApiService
from urllib.request import build_opener,ProxyHandler
service=ApiService(port=int(sys.argv[1]),runtime_dir=sys.argv[2])
url=service.ensure_running()
with build_opener(ProxyHandler({})).open(url+'/api/health') as r:
    print(json.dumps({'pid':r.headers['X-BankInsight-Pid']}),flush=True)
sys.stdin.buffer.read()
service.stop()
'''
    clients=[]
    try:
        for _ in range(3):
            clients.append(subprocess.Popen([sys.executable,'-u','-c',program,str(port),str(tmp_path)],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True))
        with ThreadPoolExecutor(max_workers=3) as pool:
            def read_pid(process):
                while True:
                    line=process.stdout.readline()
                    if not line:
                        raise RuntimeError('Le client de test a quitté avant de répondre.')
                    if line.startswith('{'):
                        return json.loads(line)['pid']
            reads=[pool.submit(read_pid,p) for p in clients]
            pids=[f.result(timeout=90) for f in reads]
        assert len(set(pids))==1
    finally:
        for client in clients:
            client.stdin.close()
        for client in clients:
            client.wait(timeout=20)
    assert all(c.returncode==0 for c in clients)
