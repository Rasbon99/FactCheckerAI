#!/bin/bash
/bin/ollama serve &
SERVE_PID=$!
sleep 5
ollama pull phi3.5:latest
wait $SERVE_PID