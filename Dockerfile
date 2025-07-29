# Stage 1: Builder - for installing Python dependencies and NLTK data
FROM python:3.11-slim AS builder

# Install system build tools required for some Python packages
# (e.g., pillow, numpy often need these)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libfreetype6-dev \
      libpng-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements.txt to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir is good for build size, but ensure you also clean pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords data to a known location inside the builder stage
# We'll copy this specific directory in the final stage
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

# Stage 2: Runner - the lean, final image for production deployment
FROM python:3.11-slim

# Install runtime system dependencies if any are needed by your app
# Example: If your app uses an image manipulation library that needs specific runtime libs
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       libjpeg-dev \
#       zlib1g-dev && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the installed Python packages from the builder stage
# This significantly reduces the image size by avoiding build tools and caches.
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy the NLTK data specifically
COPY --from=builder /usr/local/share/nltk_data/ /usr/local/share/nltk_data/

# Set NLTK_DATA environment variable so your app finds the data
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy your application code
# It's important to copy app code AFTER dependencies to maximize cache hits
COPY . .

# Expose your Flask/Gunicorn port
EXPOSE 8080

# Environment variables for Flask
ENV FLASK_APP=app/main.py \
    FLASK_ENV=production

# Use Gunicorn as WSGI server
# Using sh -c for port dynamic binding is good for Elastic Beanstalk
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} app.main:app"]