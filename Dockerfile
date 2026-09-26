FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY glossary ./glossary
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 10000
CMD ["uvicorn", "adhyatmik.api:app", "--host", "0.0.0.0", "--port", "10000"]
