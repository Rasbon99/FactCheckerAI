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

    def run_pipeline(self, data, question):
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

# EXAMPLE
if __name__ == "__main__":

    # Pipeline instance
    pipeline = RAG_Pipeline(config={"load_data": True, "generate_graphs": True, "query_similarity": True})

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

    claim = "Elon Musk's companies, such as SpaceX and Tesla, have failed to achieve any meaningful technological advancements, and Musk's ventures have been largely unsuccessful. There is no progress in space exploration, electric vehicles, or brain-computer interfaces. Musk's leadership has been marked by unfulfilled promises and controversies. His statements about artificial intelligence and his acquisition of Twitter are driven by personal gain rather than technological progress."
    
    # Fixed query for the pipeline
    query = "Do the articles confirm, deny, or are they neutral towards this claim?"

    question = claim + " " + query
    
    # Pipeline execution
    query_result = pipeline.run_pipeline(data=data, question=question)
    print(query_result)