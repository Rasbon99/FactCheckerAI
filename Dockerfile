FROM python:3.10-slim

# Evita problemi di buffering nei log
ENV PYTHONUNBUFFERED=1

# Imposta la cartella di lavoro
WORKDIR /app

# Aggiorna e installa dipendenze di sistema (se necessarie)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia i file necessari
COPY requirements.txt .  

# Installa le dipendenze Python senza cache
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Copia tutto il codice (dopo aver impostato .dockerignore)
COPY . .

# Espone le porte usate dalle varie applicazioni
EXPOSE 8001 8003 8501

# Comando di default (ma viene sovrascritto da docker-compose)
CMD ["bash"]
