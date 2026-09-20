FROM python:3.11-slim

RUN pip install --no-cache-dir httpx==0.28.1 \
    && useradd --create-home --uid 10001 tabcomplete

WORKDIR /app
COPY src/tinycomplete/__init__.py /app/tinycomplete/__init__.py
COPY src/tinycomplete/playground /app/tinycomplete/playground

USER 10001:10001
EXPOSE 8765
ENTRYPOINT ["python", "-m", "tinycomplete.playground.server"]
