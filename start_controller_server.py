import uvicorn

from controller import app

#TODO In teoria morirà, se davvero dockerizziamo

uvicorn.run(app, host="0.0.0.0", port=8003)