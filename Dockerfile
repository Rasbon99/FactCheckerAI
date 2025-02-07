# Dockerfile
FROM python:3.10-slim

# Imposta la working directory
WORKDIR /app

# Copia il requirements.txt e installa le dipendenze
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia tutto il codice
COPY . .

# (Facoltativo) Espone le porte usate dalle varie applicazioni.
# Queste istruzioni sono solo documentative, la pubblicazione effettiva le gestisce docker-compose.
EXPOSE 8001 8003 8501

# Il comando di default non viene utilizzato (verrà sovrascritto da docker-compose)
CMD ["bash"]
