# ── builder: install dependencies ──────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── runtime: copy only what's needed ────────────────────────────────────────
FROM python:3.9-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py model_loader.py schemas.py ./

# Model artifact is mounted at runtime — not baked into the image
ENV MODEL_PATH=models/model.joblib
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
