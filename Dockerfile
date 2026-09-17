# Multi-stage production container for Surya OCR Streamlit Frontend
FROM python:3.11-slim as builder

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Create virtual environment and install packages
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv /opt/venv && \
    uv pip install --no-cache hatchling && \
    uv pip install --no-cache -e . && \
    uv pip install --no-cache "streamlit>=1.31.0"

# --- Production Runner Stage ---
FROM python:3.11-slim as runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source files
COPY surya ./surya
COPY static ./static
COPY signatures ./signatures
COPY pyproject.toml README.md ./

# Re-link package in editable/local mode so surya imports work seamlessly
RUN pip install --no-deps -e .

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IN_STREAMLIT=true \
    SURYA_INFERENCE_URL=http://surya-backend-gpu:8000/v1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "surya/scripts/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.fileWatcherType=none", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false", \
            "--server.maxUploadSize=100"]
