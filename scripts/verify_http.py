"""Vérification réelle du proxy, API et protocole WebSocket Streamlit."""
import argparse
import base64
import json
import os
import socket
import ssl
from urllib.parse import urlparse
import requests


def verify(base):
    base=base.rstrip('/')
    session = requests.Session()
    session.trust_env = False
    assert session.get(base + '/api/health', timeout=10).json()['status'] == 'ok'
    assert session.get(base + '/api/docs', timeout=10).status_code == 200
    assert session.get(base + '/api/openapi.json', timeout=10).status_code == 200
    assert session.get(base + '/_stcore/health', timeout=10).text == 'ok'
    assert session.get(base, timeout=10).status_code == 200
    models = session.get(base + '/api/models', timeout=10).json()
    assert len(models) >= 5
    evidence = json.load(open('reports/experiment.json', encoding='utf-8'))
    body = {'model_id':evidence['selected_id'], 'customers':[evidence['customer']], 'threshold':.5}
    prediction = session.post(base + '/api/predict', json=body, timeout=30)
    assert prediction.status_code == 200, prediction.text
    local = session.post(base + '/api/explain/local', json={'model_id':body['model_id'],'customer':body['customers'][0]}, timeout=240)
    assert local.status_code == 200, local.text
    assert abs(local.json()['probability'] - prediction.json()['probabilities'][0]) < 1e-6
    global_result = session.post(base + '/api/explain/global', json={'model_id':body['model_id'],'sample_size':4}, timeout=240)
    assert global_result.status_code == 200, global_result.text
    parsed = urlparse(base)
    # A real RFC6455 upgrade verifies nginx forwards Streamlit WebSocket headers.
    port=parsed.port or (443 if parsed.scheme=='https' else 80)
    connection=socket.create_connection((parsed.hostname,port),timeout=10)
    if parsed.scheme=='https':
        connection=ssl.create_default_context().wrap_socket(connection,server_hostname=parsed.hostname)
    with connection as sock:
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f'GET /_stcore/stream HTTP/1.1\r\nHost: {parsed.netloc}\r\nOrigin: {base}\r\n'
                   f'Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n'
                   'Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: streamlit\r\n\r\n')
        sock.sendall(request.encode())
        response = sock.recv(4096).decode(errors='replace')
        assert '101 Switching Protocols' in response, response
    print('HTTP_PROXY_API_PREDICT_SHAP_LIME_WEBSOCKET_OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:7860')
    verify(parser.parse_args().base)
