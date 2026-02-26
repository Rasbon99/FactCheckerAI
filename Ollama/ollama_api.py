from Ollama.ollama_client import OllamaClient

from fastapi import FastAPI

ollama_app = FastAPI()

ollama_server = OllamaClient()

@ollama_app.post("/start")
def start():
    return ollama_server.start_server()

@ollama_app.post("/stop")
def stop():
    return ollama_server._stop_server()

@ollama_app.get("/status")
def status():
    return ollama_server.is_running()
    