# syntax=docker/dockerfile:1

# ---- builder: installs runtime deps into a self-contained venv ------------
FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- test-builder: same venv, plus dev deps (pytest) -----------------------
FROM builder AS test-builder
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# ---- runtime-base: slim image + only the OS libs opencv/torch need --------
# (no pip, no apt lists, no build-time files — just what's needed to run)
FROM python:3.11-slim AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# ---- test: same OS layer, but the venv that also has pytest --------------
FROM runtime-base AS test
COPY --from=test-builder /opt/venv /opt/venv
COPY . .
CMD ["python", "-m", "pytest", "-v"]

# ---- runtime: the image actually shipped/run (docker build's default) ----
FROM runtime-base AS runtime
COPY --from=builder /opt/venv /opt/venv
COPY . .
ENTRYPOINT ["python", "main.py", "--no-display"]
CMD ["--help"]
