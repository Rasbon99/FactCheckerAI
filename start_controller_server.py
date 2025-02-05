import uvicorn

from controller import app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(app, port=8003)