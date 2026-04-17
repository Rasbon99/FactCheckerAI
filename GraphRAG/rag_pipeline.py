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
            "load_data": True,
            "generate_graphs": True,
            "query_similarity": True,
        }
        if config:
            self.config.update(config)

        self.graph_manager.reset_data()

        self.graph_folder = os.getenv("GRAPHS_PATH", "data/graphs")

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

        path_graph_topics = f"{output_folder}/graph_topics.jpg"
        path_graph_entities = f"{output_folder}/graph_entities.jpg"
        path_graph_sites = f"{output_folder}/graph_sites.jpg"

        self.logger.info("Starting graph generation...")
        try:
            self.graph_manager.extract_and_save_graph(
                path_graph_topics, path_graph_entities, path_graph_sites
            )
        except Exception as e:
            self.logger.error(f"Error during graph generation: {e}")

    def query_similarity(self, query):
        """
        Executes a similarity query using the QueryEngine and tracks LLM usage metrics.

        Args:
            query (str): The query string containing the claim and verification instructions.

        Returns:
            tuple: A tuple containing:
                - str: The final verdict (e.g., Supported, Refuted, or Not Enough Info).
                - dict: A dictionary with 'total' (total tokens) and 'calls' (number of LLM requests).

        Raises:
            None: Internal exceptions are caught and logged, returning (None, 0-metrics) instead.
        """
        if not self.config.get("query_similarity", True):
            self.logger.info("Similarity query disabled by configuration.")
            return None, {"total": 0, "calls": 0}

        self.logger.info("Starting similarity query...")
        try:
            result, token_data = self.query_engine.query_similarity(query)
            self.logger.info("Similarity query completed.")
            return result, token_data
        except Exception as e:
            self.logger.error(f"Error during similarity query execution: {e}")
            return None, {"total": 0, "calls": 0}

    def run_pipeline(self, data, claim, claim_id):
        """
        Executes the entire RAG pipeline: data loading, graph generation, and fact-checking.

        Args:
            data (list): List of dictionaries containing the scraped article data.
            claim (str): The specific claim text to be verified.
            claim_id (str): The unique ID of the claim, used to organize graph assets.

        Returns:
            tuple: A tuple containing:
                - str: The AI's final verdict and reasoning (Supported/Refuted/NEI).
                - str: The file path to the folder containing the generated graph images.
                - dict: Token usage and call count metrics for efficiency evaluation.

        Raises:
            None: Internal exceptions are caught, logged, and return (None, None, 0-metrics).
        """
        self.logger.info("Starting the entire pipeline...")
        start_time = time.time()
        try:
            # Step 1: Load the data
            self.load_data(data)

            claim_graphs_folder = f"{self.graph_folder}/{claim_id}"

            if not os.path.exists(claim_graphs_folder):
                os.makedirs(claim_graphs_folder)
                self.logger.info(f"Create '{claim_graphs_folder}' folder.")

            # Step 2: Generate and save graphs
            self.generate_and_save_graphs(claim_graphs_folder)

            question = f"""
                You are a strict fact-checking assistant. 
            
                CLAIM TO EVALUATE: "{claim}"

                Based ONLY on the information provided in the retrieved articles, determine if the claim above is confirmed or refuted.

                Your response MUST follow this exact structure:
                VERDICT: [Choose ONLY one: SUPPORTS, REFUTES, or NOT ENOUGH INFO]
                REASONING: [Your detailed explanation and citations here]

                Follow these logical rules for the VERDICT:
                - If the articles confirm the claim, use 'SUPPORTS'.
                - If the articles completely contradict the claim, use 'REFUTES'.
                - If there is confusion or the articles do not mention the specific details of the claim, use 'NOT ENOUGH INFO'.

                Make sure to cite the titles of the articles that support your conclusions. Do not include any external knowledge.
            """

            # Step 3: Execute the similarity query and catch token data
            result, token_data = self.query_similarity(question)

            total_time = time.time() - start_time
            self.logger.info(
                f"Pipeline completed successfully in {total_time:.2f} seconds."
            )

            return result, claim_graphs_folder, token_data

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(
                f"Error during pipeline execution (total time: {total_time:.2f} seconds): {e}"
            )
            return None, None, {"total": 0, "calls": 0}
