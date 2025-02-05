from Ollama.ollama_client import OllamaClient

from fastapi import FastAPI

ollama_app = FastAPI()

ollama_server = OllamaClient()

@ollama_app.post("/start")
def start():
    return ollama_server.start_server()

@ollama_app.post("/stop")
def stop():
    return ollama_server.stop_server()

@ollama_app.get("/status")
def status():
    return {"running": ollama_server.is_running()}
    