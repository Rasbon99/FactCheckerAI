import uvicorn

from controller import app
from log import Logger

logger = Logger("ControllerServer").get_logger()

try:
    uvicorn.run(app, host="0.0.0.0", port=8003)
except Exception as e:
    logger.exception(f"Controller server failed: {e}")
    raise