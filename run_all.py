import subprocess
import sys
from pathlib import Path
from log import Logger

ROOT = Path(__file__).parent
logger = Logger("RunAll").get_logger()

services = [
    #[sys.executable, "start_neo4j_server.py"],
    #[sys.executable, "start_llamacpp_server.py"],
    [sys.executable, "start_controller_server.py"],
    [sys.executable, "start_backend_server.py"],
    [sys.executable, "-m", "streamlit", "run", "Dashboard/dashboard.py"],
]

for service in services:
    try:
        subprocess.Popen(
            service,
            cwd=ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        logger.info(f"Started service: {' '.join(service)}")
    except Exception as e:
        logger.exception(
            f"Failed to start service {' '.join(service)}: {e}"
        )
        raise

logger.info("All service processes started.")

print("All services started.")