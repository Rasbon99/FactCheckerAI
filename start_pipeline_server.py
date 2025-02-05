import uvicorn

from pipeline_api import pipeline_app

uvicorn.run(pipeline_app, port=8001)
