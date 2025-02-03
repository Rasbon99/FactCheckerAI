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


claim = Claim("The earth is flat")
answer = Answer(claim_id=claim.id, answer="The earth is not flat", image=Image.open(os.getenv('AI_IMAGE_UI')))
# example of sources str(uuid.uuid4()), self.id, data['title'], data['url'], data['site'],
#                  data['body'], data['score'], data['topic'], str(data['entities']
data = [{
    'title': "Mi piace la zucchina",
    'url': "url",
    'site': "site",
    'body': "body",
    'score': 100,
    'topic': "topic",
    'entities': ["entity1", "entity2"]
},
    {'title': "Mi piace la zucchina",
    'url': "url",
    'site': "site",
    'body': "body",
    'score': 100,
    'topic': "topic",
    'entities': ["entity1", "entity2"]
}]
claim.add_sources(data)


if __name__=="__main__":
    
    logger = Logger("DashboardLogger", log_file=os.getenv('LOG_FILE')).get_logger()
    logger.info(f"Running command: streamlit run")
    os.system("streamlit run ./Dashboard/dashboard.py --server.runOnSave=true")