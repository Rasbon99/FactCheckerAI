#!/bin/bash
# Avvia il servizio Ollama in background
/bin/ollama serve &
SERVE_PID=$!
# Attendi qualche secondo per essere sicuro che il servizio sia attivo
# (puoi eventualmente sostituire con un controllo di readiness se disponibile)
sleep 5
# Esegui il pull del modello desiderato
ollama pull phi3.5:latest
# Rimani in attesa del processo di ollama serve (mantiene attivo lo script)
wait $SERVE_PID