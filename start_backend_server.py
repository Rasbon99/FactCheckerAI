import uvicorn

from backend import backend_app

uvicorn.run(backend_app, host="0.0.0.0", port=8001)
