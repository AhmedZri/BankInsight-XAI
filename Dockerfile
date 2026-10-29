FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends nginx libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 student && chown -R student:student /app
USER student
ENV MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
EXPOSE 7860
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
CMD ["python", "deploy/start.py"]
