FROM python:3.10-slim

# Avoid buffering issues in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Update and install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the required files
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all the code (after setting up .dockerignore)
COPY . .

# Expose the ports used by the various applications
EXPOSE 8001 8003 8501

# Default command (overridden by docker-compose)
CMD ["bash"]