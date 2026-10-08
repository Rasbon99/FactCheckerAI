import os

from log import Logger

if __name__=="__main__":
    
    logger = Logger("DashboardDriver").get_logger()
    logger.info(f"Running command: streamlit run")

    exit_code = os.system("streamlit run ./Dashboard/dashboard.py --server.runOnSave=true")

    if exit_code != 0:
        logger.error(f"Streamlit exited with code {exit_code}")
        raise RuntimeError(f"Streamlit failed with exit code {exit_code}")