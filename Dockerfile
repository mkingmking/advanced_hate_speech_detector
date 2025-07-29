# Use a slim Python base
FROM python:3.11-slim

# 1. Set workdir
WORKDIR /app

# 2. Copy only requirements first (for caching)
COPY requirements.txt /app/

# 3. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app
COPY . /app

# 5. Expose Flask’s port
EXPOSE 5000

# 6. Set environment variable for Flask
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production

# 7. Run the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
