import uvicorn

from backend import backend_app
from log import Logger

logger = Logger("BackendServer").get_logger()

try:
    uvicorn.run(backend_app, host="0.0.0.0", port=8001)
except Exception as e:
    logger.exception(f"Backend server failed: {e}")
    raise