# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Runtime libs opencv-python/torch need even with no display attached
# (headless container — the app always runs with --no-display here).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["python", "-m", "pytest", "-v"]

FROM base AS runtime
ENTRYPOINT ["python", "main.py", "--no-display"]
CMD ["--help"]
