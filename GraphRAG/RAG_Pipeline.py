import dotenv
from GraphManager import GraphManager
from QueryEngine import QueryEngine
import time
import sys
import os

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

class RAG_Pipeline:
    def __init__(self, env_file="RAGkey.env", graph_config=None, query_config=None, config=None):
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

    def load_data(self, csv_path):
        """
        Carica i dati nel grafo tramite il GraphManager.
        """
        if not self.config.get("load_data", True):
            self.logger.info("Caricamento dei dati disabilitato dalla configurazione.")
            return

        self.logger.info("Avvio del caricamento dei dati...")
        try:
            self.graph_manager.load_data(csv_path)
            self.logger.info("Dati caricati con successo.")
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento dei dati: {e}")
            raise

    def generate_and_save_graphs(self, output_file_topic="graph_topic.jpg", output_file_published="graph_published.jpg"):
        """
        Genera e salva i grafi tramite il GraphManager.
        """
        if not self.config.get("generate_graphs", True):
            self.logger.info("Generazione dei grafi disabilitata dalla configurazione.")
            return

        self.logger.info("Avvio della generazione dei grafi...")
        try:
            self.graph_manager.extract_and_save_graph(output_file_topic, output_file_published)
            self.logger.info("Grafi generati e salvati con successo.")
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

    def run_pipeline(self, csv_path, query):
        """
        Esegui l'intera pipeline: carica i dati, genera i grafi e rispondi alla query.
        """
        self.logger.info("Avvio dell'intera pipeline...")
        start_time = time.time()  # Inizio misurazione del tempo
        try:
            # Step 1: Carica i dati
            self.load_data(csv_path)

            # Step 2: Genera e salva i grafi
            self.generate_and_save_graphs()

            # Step 3: Esegui la query di similarità
            result = self.query_similarity(query)

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

    # Percorso al file CSV e query di similarità
    csv_path = "https://raw.githubusercontent.com/dcarpintero/generative-ai-101/main/dataset/synthetic_articles.csv"
    query = "Which articles discuss how AI might affect our daily life? Include the article titles and abstracts."

    # Esegui la pipeline
    query_result = pipeline.run_pipeline(csv_path=csv_path, query=query)
    print(query_result)