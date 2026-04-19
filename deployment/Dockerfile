FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy judicial project assets
COPY . .

# Hugging Face Spaces default port
EXPOSE 7860

# Launch the Judicial Discovery Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
