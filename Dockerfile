# Stage 1: Builder - for installing Python dependencies and NLTK data
FROM python:3.11-slim AS builder

# Install system build tools required for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libfreetype6-dev \
      libpng-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

# Stage 2: Runner - the lean, final image for production deployment
FROM python:3.11-slim

# CRITICAL FIX: Add /usr/local/bin to the PATH in the final stage
# This ensures that executables installed by pip (like gunicorn) are found.
ENV PATH="/usr/local/bin:$PATH"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

COPY --from=builder /usr/local/share/nltk_data/ /usr/local/share/nltk_data/

ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

EXPOSE 8080

ENV FLASK_APP=app/main.py \
    FLASK_ENV=production

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} app.main:app"]