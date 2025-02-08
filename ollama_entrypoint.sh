#!/bin/bash
echo starting pulling...

ollama serve

# Scarica il modello desiderato
ollama pull phi3.5:latest

# Avvia il servizio Ollama come processo principale
exec /bin/ollama serve
