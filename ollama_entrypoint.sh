#!/bin/bash
# Start the Ollama service in the background
/bin/ollama serve &
SERVE_PID=$!
sleep 5
# Pull the desired model
ollama pull phi3.5:latest
# Wait for the ollama serve process
wait $SERVE_PID