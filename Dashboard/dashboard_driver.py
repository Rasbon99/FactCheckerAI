import os
from log import Logger

if __name__=="__main__":
    logger = Logger("DashboardLogger", log_file="logger.log").get_logger()
    logger.info(f"Running command: streamlit run")
    os.system("streamlit run ./Dashboard/dashboard.py")