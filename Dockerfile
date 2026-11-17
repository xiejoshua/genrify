# python:3.12-slim: the base image. It has Python 3.12 and little else. 3.12 has prebuilt wheels for numpy and scikit-learn, so pip does not compile anything. "slim" keeps the image small. A bare "python" tag would drift to a new version later and could break the pinned packages
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# requirements.txt first, on its own: Docker caches each step, so editing app code does not reinstall every package. The ./ means "into /app"
COPY requirements.txt ./
RUN pip install -r requirements.txt

# backend: the folder with app/ and models/. It must be copied in, because the container only contains what you COPY. If models/ is missing, the server fails at startup. The destination ./backend matches the --app-dir below
COPY backend ./backend

RUN useradd --create-home appuser
USER appuser

# 0.0.0.0: listen on every network interface. 127.0.0.1 would only accept connections from inside the container, so your browser, curl, and Render could never reach it. 8000 is only the fallback: Render sets $PORT, and the shell expands it here
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
