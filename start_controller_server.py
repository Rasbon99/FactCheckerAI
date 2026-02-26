import uvicorn

from controller import app

uvicorn.run(app, host="0.0.0.0", port=8003)