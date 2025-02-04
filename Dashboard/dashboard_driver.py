import os
from log import Logger
from dotenv import load_dotenv
import sys
from PIL import Image

# Test visualizzazione storico chat nella dashboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.data_entities import Claim
from Database.data_entities import Answer

load_dotenv(dotenv_path='key.env')

if __name__=="__main__":
    
    logger = Logger("DashboardLogger", log_file=os.getenv('LOG_FILE')).get_logger()
    logger.info(f"Running command: streamlit run")
    os.system("streamlit run ./Dashboard/dashboard.py --server.runOnSave=true")