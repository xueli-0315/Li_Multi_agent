FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV PYTHONPATH=/app/src
ENV BACKTEST_ENGINE=qlib

RUN pip install --upgrade pip && pip install -e .

CMD ["python", "main.py", "--loop-count", "1"]
