FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY app.py .
COPY model.h5 .
COPY requirements.txt .

# Install dependencies
# FIX: TensorFlow 2.14.0 requires numpy < 2.0
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

