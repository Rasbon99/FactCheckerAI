import uvicorn

from backend import backend_app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(backend_app, port=8001)
