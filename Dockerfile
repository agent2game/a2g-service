FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY decision_service.py monitor.py dashboard.html ./
RUN useradd --uid 10001 a2g && mkdir /data && chown a2g /data
USER a2g
ENV A2G_HOST=0.0.0.0 A2G_DB=/data/a2g.db PYTHONUNBUFFERED=1
EXPOSE 8787
VOLUME /data
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('A2G_PORT', '8787'), timeout=2)"
CMD ["python", "decision_service.py"]
