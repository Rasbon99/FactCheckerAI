import uvicorn

from Ollama.ollama_api import ollama_app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(ollama_app, port=8000)

