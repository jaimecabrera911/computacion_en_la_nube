# ---------------------------------------------------------------------------
# Imagen de producción — Sistema RAG Cloud Serverless en Salud (ACA N° 1)
#
#   docker build -t computacion-nube:1.0.0 .
#   docker run -p 8501:8501 --env-file .env computacion-nube:1.0.0
#
# Notas de diseño:
#   * Etapa única (la app necesita torch/sentence-transformers en runtime).
#   * El modelo de embeddings se descarga en BUILD, no en runtime: evita que
#     el primer usuario espere la descarga de ~500 MB y mantiene la imagen
#     reproducible aunque huggingface.co esté intermitente.
#   * uv se usa como instalador: respeta uv.lock y reproduce el entorno local.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf-cache \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# curl  → healthcheck del contenedor
# build-essential no se necesita: torch, sentence-transformers, numpy y pandas
# publican ruedas precompiladas para cp312 en manylinux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 1) Capa de dependencias (cacheable: solo cambia si cambian los manifests)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Código y artefactos del proyecto
COPY . .
RUN uv sync --frozen --no-dev \
    && chmod +x scripts/*.py 2>/dev/null || true

# 3) Descarga del modelo de embeddings (una sola vez, en build)
ARG EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('${EMBEDDING_MODEL_NAME}'); \
print('embeddings ok ->', m.get_sentence_embedding_dimension(), 'dims')"

# Usuario sin privilegios
RUN useradd --create-home --shell /bin/bash rag \
    && chown -R rag:rag /app /opt/hf-cache
USER rag

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# $PORT permite desplegar tal cual en PaaS que inyectan el puerto
# (Cloud Run, Render, Railway, Fly.io); en local/compose cae a 8501.
CMD ["sh", "-c", "exec uv run streamlit run app/app_sustentacion.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
