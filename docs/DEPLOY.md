# Despliegue del proyecto — Sistema RAG Cloud Serverless en Salud (ACA N° 1)

Este documento explica cómo pasar la app de sustentación (Streamlit) de tu
máquina local a un servidor real. Hay **tres rutas**, de menor a mayor effort:

| # | Ruta | Cuándo usarla | Costo típico |
|---|------|---------------|--------------|
| A | **VPS Ubuntu** (systemd + nginx + TLS) | Exposición estable, control total, tesis con infraestructura propia | US$ 4–6/mes (Hetzner/DigitalOcean) |
| B | **Docker Compose en un VPS** | Igual que A pero reproducible y portable | igual que A |
| C | **PaaS serverless** (Render, Cloud Run, Fly, App Runner) | Sin administrar servidor, escalado automático | free tier / pay-per-use |

---

## 0. Requisitos previos (todos)

```bash
# 1) El código en GitHub (ya está: origin -> github.com/jaimecabrera911)
git push origin main

# 2) Credenciales de Qdrant Cloud en el servidor, NO en el repo.
#    Se leen de .env (dotenv) o de las variables de entorno del PaaS.
#    Variables requeridas:
#      QDRANT_URL=https://xxxx.us-east4-0.gcp.cloud.qdrant.io:6333
#      QDRANT_API_KEY=...
#      QDRANT_COLLECTION_NAME=clinical_heart_disease_rag
#      EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 3) La app NO regenera el índice: asume que la colección ya existe
#    en Qdrant Cloud con los 349 vectores. Para reconstruirla:
#    uv run rag pipeline   (o `make pipeline`)
```

> ⚠️ **Nunca subas el `.env` a GitHub.** Ya está en `.gitignore` y en
> `.dockerignore`. En los PaaS usa *Secrets/Environment Variables*, no
> variables "normales" de build.

---

## Ruta A — VPS Ubuntu (recomendada para la sustentación)

### A.1 Preparar el servidor

```bash
# Ubuntu 24.04 LTS, 1 vCPU / 2 GB RAM mínimo (torch + modelo ≈ 1.5 GB en RAM)
ssh root@TU_IP

apt update && apt install -y python3.12 python3.12-venv git nginx curl
curl -LsSf https://astral.sh/uv/install.sh | sh   # instalador de uv
```

### A.2 Desplegar el código y el entorno

```bash
useradd -m -s /bin/bash rag
mkdir -p /opt/computacion-en-la-nube
chown rag:rag /opt/computacion-en-la-nube

sudo -u rag git clone https://github.com/jaimecabrera911/computacion_en_la_nube.git \
    /opt/computacion-en-la-nube
cd /opt/computacion-en-la-nube

sudo -u rag -H bash -c 'uv sync'         # crea .venv con uv.lock
sudo -u rag -H bash -c 'uv run python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer(\"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\")
print(\"modelo descargado\")"'
```

### A.3 Secretos

```bash
sudo -u rag -H bash -c 'cat > /opt/computacion-en-la-nube/.env <<EOF
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=clinical_heart_disease_rag
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EOF'
sudo chmod 600 /opt/computacion-en-la-nube/.env
```

### A.4 Servicio systemd

```bash
sudo cp /opt/computacion-en-la-nube/deploy/computacion-nube.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now computacion-nube
sudo systemctl status computacion-nube
curl -s http://127.0.0.1:8501/_stcore/health   # -> ok
```

### A.5 nginx + HTTPS (Let's Encrypt)

```bash
sudo cp /opt/computacion-en-la-nube/deploy/nginx.conf /etc/nginx/sites-available/computacion-nube
sudo sed -i 's/rag.tudominio.com/TU_DOMINIO/g' /etc/nginx/sites-available/computacion-nube
sudo ln -s /etc/nginx/sites-available/computacion-nube /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Certificado gratuito (renovación automática vía timer systemd)
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d TU_DOMINIO
```

### A.6 Firewall

```bash
sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable
```

### A.7 Actualizar el servidor cuando subes cambios

```bash
sudo -u rag -H bash -c 'cd /opt/computacion-en-la-nube && git pull && uv sync'
sudo systemctl restart computacion-nube
```

---

## Ruta B — Docker Compose (reproducible)

```bash
# En el servidor (o en tu PC para probar)
git clone https://github.com/jaimecabrera911/computacion_en_la_nube.git
cd computacion_en_la_nube

# Credenciales en el servidor (nunca en el repo)
nano .env      # pega QDRANT_URL, QDRANT_API_KEY, ...

docker compose up -d --build
docker compose logs -f app
curl -s http://localhost:8501/_stcore/health   # -> ok
```

Notas:
- La imagen pesa ~3–4 GB (torch + modelo embebido en build). El primer
  `build` tarda 10–20 min; después, la cache de capas lo acelera.
- `hf-cache` es un volumen nombrado: el modelo **no** se redescarga en cada
  arranque.
- Si expones el puerto 8501 directamente, recuerda cerrar el 8501 en el firewall
  y usar nginx por delante (igual que en la ruta A).

---

## Ruta C — PaaS serverless (sin servidor que administrar)

Todas estas plataformas detectan el `Dockerfile` automáticamente.

### C.1 Google Cloud Run (la más "cloud-native")

```bash
gcloud run deploy computacion-nube \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars QDRANT_COLLECTION_NAME=clinical_heart_disease_rag,EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --set-secrets QDRANT_URL=qdrant-url:latest,QDRANT_API_KEY=qdrant-key:latest
```

- `PORT` lo inyecta Cloud Run → el `CMD` del Dockerfile lo respeta.
- Escala a cero cuando no hay tráfico (coste ≈ 0) y enciende una instancia
  en el arranque en frío (≈ 30 s la primera vez).

### C.2 Render / Railway / Fly.io

- **Render**: servicio *Web Service* → Environment *Docker*; pega las
  variables en *Environment Variables*; `Health Check Path` = `/_stcore/health`.
- **Railway**: `railway init && railway up` con el Dockerfile; define las
  variables en el dashboard; Railway detecta `$PORT` automáticamente.
- **Fly.io**: `fly launch --no-deploy` y luego `fly deploy`; secretos con
  `fly secrets set QDRANT_API_KEY=...`.

### C.3 AWS App Runner / Azure App Service

- **App Runner**: nueva *Service* → *Source* = el repo GitHub → *Deployment*
  = *Container* → *Runtime environment* = Amazon Linux 2023, 2 vCPU / 4 GB
  (el modelo no cabe cómodo en 1 GB).
- **Azure App Service**: *Web App* → *Container Apps* → imagen desde ACR o
  desde GitHub Actions; `WEBSITES_PORT=8501` en la configuración.

---

## Checklist de verificación tras desplegar

```bash
# 1) La app responde
curl -s https://TU_DOMINIO/_stcore/health          # -> ok

# 2) La app sirve la interfaz
curl -s https://TU_DOMINIO/ | grep -q "RAG Cloud" && echo "UI ok"

# 3) La conexión a Qdrant Cloud funciona (la demo "Demo RAG en Vivo" busca
#    en tiempo real). Si falla: credenciales o allowlist de IP de Qdrant.
# 4) TLS válido
curl -sI https://TU_DOMINIO | head -1                # HTTP/2 200
```

---

## Variables de entorno de la app (referencia)

| Variable | ¿Requerida? | Descripción |
|----------|--------------|-------------|
| `QDRANT_URL` | en producción | Endpoint del clúster Qdrant Cloud |
| `QDRANT_API_KEY` | en producción | API key del clúster |
| `QDRANT_COLLECTION_NAME` | no | Colección por defecto: `clinical_heart_disease_rag` |
| `EMBEDDING_MODEL_NAME` | no | Modelo SBERT multilingüe (384 dims) |
| `PORT` | solo PaaS | Puerto inyectado por la plataforma |

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---------|-------|----------|
| La UI carga pero no responde | WebSockets bloqueados por el proxy | En nginx añade `Upgrade`/`Connection` (ya está en `deploy/nginx.conf`) |
| `403` de Qdrant Cloud | IP del servidor no permitida o key expirada | Revisa la API key y la allowlist en cloud.qdrant.io |
| La app tarda 2 min al abrir | Descarga del modelo en runtime | Pre-descárgalo (build de Docker) o `HF_HOME` persistente |
| `Out of memory` | 1 GB de RAM no alcanza para torch + MiniLM | Sube a 2 GB mínimo (4 GB en PaaS) |
| Pantalla en blanco tras editar | `fileWatcherType = "none"` en `.streamlit/config.toml` | Reinicia el servicio: `systemctl restart computacion-nube` |
