import os
import sys
import time

import dotenv

from graph_manager import GraphManager
from query_engine import QueryEngine

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

class RAG_Pipeline:
    def __init__(self, env_file="RAGkey.env", config=None):
        # Carica variabili d'ambiente
        dotenv.load_dotenv(env_file, override=True)

        # Logger
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Configura il gestore dei grafi
        self.graph_manager = GraphManager(env_file)

        # Configura il motore di query
        self.query_engine = QueryEngine(env_file)

        # Configurazione personalizzabile
        self.config = {
            "load_data": True,             # Abilita/disabilita il caricamento dei dati
            "generate_graphs": True,       # Abilita/disabilita la generazione dei grafi
            "query_similarity": True       # Abilita/disabilita la query di similarità
        }
        if config:
            self.config.update(config)

    def load_data(self, data):
        """
        Carica i dati nel grafo tramite il GraphManager.
        """
        if not self.config.get("load_data", True):
            self.logger.info("Caricamento dei dati disabilitato dalla configurazione.")
            return

        self.logger.info("Avvio del caricamento dei dati...")
        try:
            self.graph_manager.load_data(data)
            self.logger.info("Dati caricati con successo.")
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento dei dati: {e}")
            raise

    def generate_and_save_graphs(self, output_file_topic="graph_topic.jpg", output_file_entity="graph_entity.jpg", output_file_site="graph_site.jpg"):
        """
        Genera e salva i grafi tramite il GraphManager.
        """
        if not self.config.get("generate_graphs", True):
            self.logger.info("Generazione dei grafi disabilitata dalla configurazione.")
            return

        self.logger.info("Avvio della generazione dei grafi...")
        try:
            self.graph_manager.extract_and_save_graph(output_file_topic, output_file_entity, output_file_site)
        except Exception as e:
            self.logger.error(f"Errore durante la generazione dei grafi: {e}")
            raise

    def query_similarity(self, query):
        """
        Esegue una query di similarità tramite il QueryEngine.
        """
        if not self.config.get("query_similarity", True):
            self.logger.info("Query di similarità disabilitata dalla configurazione.")
            return None

        self.logger.info("Avvio della query di similarità...")
        try:
            result = self.query_engine.query_similarity(query)
            self.logger.info("Query di similarità completata.")
            return result
        except Exception as e:
            self.logger.error(f"Errore durante l'esecuzione della query di similarità: {e}")
            return None

    def run_pipeline(self, data, question):
        """
        Esegui l'intera pipeline: carica i dati, genera i grafi e rispondi alla query.
        """
        self.logger.info("Avvio dell'intera pipeline...")
        start_time = time.time()  # Inizio misurazione del tempo
        try:
            # Step 1: Carica i dati
            self.load_data(data)

            # Step 2: Genera e salva i grafi
            self.generate_and_save_graphs()

            # Step 3: Esegui la query di similarità
            result = self.query_similarity(question)

            # Calcola il tempo totale di esecuzione
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completata con successo in {total_time:.2f} secondi.")

            return result
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Errore durante l'esecuzione della pipeline (tempo totale: {total_time:.2f} secondi): {e}")
            return None

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un'istanza della pipeline
    pipeline = RAG_Pipeline(config={"load_data": True, "generate_graphs": True, "query_similarity": True})

    # Dati di esempio
    data = [
        {
            "TITLE": "Elon Musk's SpaceX Launches Another Starship Test Flight",
            "URL": "https://example.com/spacex-launches-another-test-flight",
            "SITE": "SpaceNews",
            "BODY": "SpaceX, the private space company led by Elon Musk, successfully launched its latest Starship test flight. This test is a significant milestone in SpaceX's plans to send humans to Mars. Musk has promised that Starship will be key to achieving this ambitious goal.",
            "ENTITY": ["Elon Musk", "SpaceX", "Starship", "Mars"],
            "TOPIC": ["Space Exploration", "Technology", "SpaceX"]
        },
        {
            "TITLE": "Tesla's New Electric Vehicle Features Unveiled by Elon Musk",
            "URL": "https://example.com/tesla-electric-vehicle-features",
            "SITE": "TechCrunch",
            "BODY": "Elon Musk revealed the new features for Tesla's latest electric vehicle model, which includes an improved battery life and autonomous driving capabilities. This announcement was made during a live event streamed worldwide. Tesla continues to lead the electric car revolution under Musk's leadership.",
            "ENTITY": ["Elon Musk", "Tesla", "Electric Vehicle", "Autonomous Driving"],
            "TOPIC": ["Electric Vehicles", "Technology", "Tesla"]
        },
        {
            "TITLE": "Elon Musk Talks About Artificial Intelligence in Latest Interview",
            "URL": "https://example.com/musk-interview-ai",
            "SITE": "AI News",
            "BODY": "In a recent interview, Elon Musk discussed the future of artificial intelligence and its potential risks. Musk, who has been a vocal critic of unregulated AI development, shared his concerns about the ethical implications of AI systems becoming more advanced.",
            "ENTITY": ["Elon Musk", "Artificial Intelligence", "AI", "Technology"],
            "TOPIC": ["Artificial Intelligence", "Technology", "Ethics"]
        },
        {
            "TITLE": "Musk's Neuralink Successfully Tests Brain-Computer Interface",
            "URL": "https://example.com/neuralink-brain-computer-interface",
            "SITE": "TechRadar",
            "BODY": "Neuralink, the neurotechnology company founded by Elon Musk, has successfully tested a new brain-computer interface. The test demonstrated how the device could allow paralyzed individuals to control devices with their minds, marking a huge step forward in neuroscience.",
            "ENTITY": ["Elon Musk", "Neuralink", "Brain-Computer Interface", "Neuroscience"],
            "TOPIC": ["Neurotechnology", "Technology", "Healthcare"]
        },
        {
            "TITLE": "Elon Musk's Twitter Acquisition: What It Means for the Social Media Landscape",
            "URL": "https://example.com/musk-twitter-acquisition",
            "SITE": "Business Insider",
            "BODY": "Elon Musk's acquisition of Twitter has sparked intense debate over the future of social media. Some believe that Musk's leadership will bring more freedom of speech to the platform, while others are concerned about the potential for increased misinformation and polarization.",
            "ENTITY": ["Elon Musk", "Twitter", "Social Media", "Freedom of Speech"],
            "TOPIC": ["Social Media", "Business", "Technology"]
        }
    ]

    # Claim di esempio
    claim = "Elon Musk's companies, such as SpaceX and Tesla, have failed to achieve any meaningful technological advancements, and Musk's ventures have been largely unsuccessful. There is no progress in space exploration, electric vehicles, or brain-computer interfaces. Musk's leadership has been marked by unfulfilled promises and controversies. His statements about artificial intelligence and his acquisition of Twitter are driven by personal gain rather than technological progress."
    
    # Query fissa per la pipeline
    query = "Do the articles confirm, deny, or are they neutral towards this claim?"

    question = claim + " " + query
    
    # Esegui la pipeline
    query_result = pipeline.run_pipeline(data=data, question=question)
    print(query_result)