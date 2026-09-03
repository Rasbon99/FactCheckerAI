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

# 1. Upgrade pip
# 2. FORCE install CPU-only PyTorch to prevent 3GB image bloat
# 3. Install the rest of the requirements
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy all the code (after setting up .dockerignore)
COPY . .

# Expose the ports used by the various applications
EXPOSE 8001 8003 8501

# Default command (overridden by docker-compose)
CMD ["bash"]