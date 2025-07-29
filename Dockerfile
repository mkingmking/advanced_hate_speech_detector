# Dockerfile

# 1) Use slim Python base
FROM python:3.11-slim

# 2) Install system build tools (gcc, headers) & cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libfreetype6-dev \
      libpng-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 3) Create and set workdir
WORKDIR /app

# 4) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords

# 5) Copy app code
COPY . .

# 6) Expose your Flask/Gunicorn port
EXPOSE 8080

# 7) Env vars
ENV FLASK_APP=app/main.py \
    FLASK_ENV=production

# 8) Use Gunicorn as WSGI server
#CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} app.main:app"]
