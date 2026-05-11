# Use a lightweight, stable Python image
FROM python:3.10-slim

# Install system dependencies required by OpenCV (used inside ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and model weights into the container
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Launch the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]