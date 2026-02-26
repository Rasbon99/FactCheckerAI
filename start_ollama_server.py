import uvicorn

from Ollama.ollama_api import ollama_app

uvicorn.run(ollama_app, host="0.0.0.0", port=8000)

