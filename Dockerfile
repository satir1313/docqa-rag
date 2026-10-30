FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install . && useradd -m app && chown -R app /app
USER app
# Bake the index at build time; mount a volume and re-run `docqa ingest` to use your own docs.
RUN docqa ingest
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["docqa", "serve", "--host", "0.0.0.0", "--port", "8000"]
