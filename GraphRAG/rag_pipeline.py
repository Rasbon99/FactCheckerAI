import dotenv
import time
import os

from GraphRAG.graph_manager import GraphManager
from GraphRAG.query_engine import QueryEngine

from log import Logger

class RAG_Pipeline:
    def __init__(self, env_file="key.env", config=None):
        """
        Initializes the RAG Pipeline, setting up environment variables, logging, graph manager, and query engine.

        Args:
            env_file (str): Path to the .env file containing configuration settings.
            config (dict, optional): Custom configuration to override default settings (load_data, generate_graphs, query_similarity).
        
        Raises:
            KeyError: If required environment variables are missing.
        """
        dotenv.load_dotenv(env_file, override=True)

        # Logger
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Configures the GraphManager
        self.graph_manager = GraphManager(env_file)

        # Configures the QueryEngine
        self.query_engine = QueryEngine(env_file)

        # Customizable configuration
        self.config = {
            "load_data": True,             # Enables/disables data loading
            "generate_graphs": True,       # Enables/disables graph generation
            "query_similarity": True       # Enables/disables similarity queries
        }
        if config:
            self.config.update(config)
        
        self.graph_manager.reset_data()

        self.graph_folder = os.getenv("ASSET_PATH")

        if not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)
            self.logger.info(f"Create '{self.graph_folder}' folder.")

    def load_data(self, data):
        """
        Loads the provided data into the graph via the GraphManager.

        Args:
            data (any): The data to be loaded into the graph.

        Raises:
            Exception: If there is an error during the data loading process.
        """
        if not self.config.get("load_data", True):
            self.logger.info("Data loading disabled by configuration.")
            return

        self.logger.info("Starting data loading...")
        try:
            self.graph_manager.load_data(data)
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error during data loading: {e}")
            raise

    def generate_and_save_graphs(self, output_folder):
        """
        Generates and saves graphs using the GraphManager.

        Args:
            output_file_topic (str): The file name to save the topic graph.
            output_file_entity (str): The file name to save the entity graph.
            output_file_site (str): The file name to save the site graph.

        Raises:
            Exception: If there is an error during graph generation.
        """
        if not self.config.get("generate_graphs", True):
            self.logger.info("Graph generation disabled by configuration.")
            return
        
        path_graph_topics=f"{output_folder}/graph_topics.jpg"
        path_graph_entities=f"{output_folder}/graph_entities.jpg"
        path_graph_sites=f"{output_folder}/graph_sites.jpg"

        self.logger.info("Starting graph generation...")
        try:
            self.graph_manager.extract_and_save_graph(path_graph_topics, path_graph_entities, path_graph_sites)
        except Exception as e:
            self.logger.error(f"Error during graph generation: {e}")

    def query_similarity(self, query):
        """
        Executes a similarity query using the QueryEngine.

        Args:
            query (str): The query string to be executed for similarity-based retrieval.
        
        Raises:
            Exception: If there is an error during the similarity query execution.
        
        Returns:
            str: The result of the similarity query, or None if the query is disabled or an error occurs.
        """
        if not self.config.get("query_similarity", True):
            self.logger.info("Similarity query disabled by configuration.")
            return None

        self.logger.info("Starting similarity query...")
        try:
            result = self.query_engine.query_similarity(query)
            self.logger.info("Similarity query completed.")
            return result
        except Exception as e:
            self.logger.error(f"Error during similarity query execution: {e}")
            return None

    def run_pipeline(self, data, claim, claim_id):
        """
        Executes the entire pipeline: load data, generate graphs, and respond to the query.

        Args:
            data (any): The data to be loaded into the graph.
            question (str): The question to be asked in the similarity query.
    
        Raises:
            Exception: If there is an error during any step of the pipeline.
        
        Returns:
            str: The result of the similarity query, or None if an error occurs.
        """
        self.logger.info("Starting the entire pipeline...")
        start_time = time.time()  # Start time measurement
        try:
            # Step 1: Load the data
            self.load_data(data)

            claim_graphs_folder = f"{self.graph_folder}/{claim_id}"

            if not os.path.exists(claim_graphs_folder):
                os.makedirs(claim_graphs_folder)
                self.logger.info(f"Create '{claim_graphs_folder}' folder.")

            # Step 2: Generate and save graphs
            self.generate_and_save_graphs(claim_graphs_folder)
            
            # Fixed query for the pipeline
            query = """Based on the information provided in the articles, determine if the claim is confirmed or refuted. 
                        - If the articles confirm the claim, validate it.
                        - If the articles completely contradict the claim or present completely different information, consider it false.
                        - If there is confusion because some articles confirm the claim while others deny it, refrain from giving an answer.
                        Additional guidelines:
                        - Keep in mind that some information may not be available in all articles, and some articles may cover only part of the claim. In such cases, evaluate the available information in each article to decide whether the claim should be accepted or not.
                        - Do not provide a "partially confirmed" or "partially refuted" response. The decision must be either to confirm or refute the claim, or to refrain from answering if the evidence is conflicting.
                        
                        Make sure to cite the titles of the articles that support your conclusions. 
                        Only use the information available in the articles, and do not include any external knowledge.
                    """
            question = "Claim: \"" + claim + "\" " + query
            
            # Step 3: Execute the similarity query
            result = self.query_similarity(question)

            # Calculate total execution time
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds.")

            return result, claim_graphs_folder
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error during pipeline execution (total time: {total_time:.2f} seconds): {e}")
            return None, None