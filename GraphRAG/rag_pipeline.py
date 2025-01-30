import time

import dotenv

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

    def generate_and_save_graphs(self, output_file_topic="graph_topic.jpg", output_file_entity="graph_entity.jpg", output_file_site="graph_site.jpg"):
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

        self.logger.info("Starting graph generation...")
        try:
            self.graph_manager.extract_and_save_graph(output_file_topic, output_file_entity, output_file_site)
        except Exception as e:
            self.logger.error(f"Error during graph generation: {e}")
            raise

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

    def run_pipeline(self, data, claim):
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

            # Step 2: Generate and save graphs
            self.generate_and_save_graphs()
            
            # Fixed query for the pipeline
            query = "Do the articles confirm, deny, or are they neutral towards this claim? Cite the titles"
            question = claim + " " + query
            
            # Step 3: Execute the similarity query
            result = self.query_similarity(question)

            # Calculate total execution time
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds.")

            return result
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error during pipeline execution (total time: {total_time:.2f} seconds): {e}")
            return None