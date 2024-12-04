# Use a lightweight Python base image
FROM python:3.9-slim

# Install system dependencies for OpenCV and Poppler (required for pdf2image)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /project

# Copy the current directory contents into the container
COPY . /project

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to execute the Python script
CMD ["python", "image_manpliation_enhancement.py"]