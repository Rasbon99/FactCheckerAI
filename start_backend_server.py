import uvicorn

from backend import backend_app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(backend_app, host="0.0.0.0", port=8001)
