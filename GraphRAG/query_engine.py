import os
import subprocess
import time
import platform
import psutil
import socket

import dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from log import Logger

class QueryEngine:
    def __init__(self, env_file="key.env", index_name="articles"):
        """
        Initializes the QueryEngine by setting up the environment variables, models, and Neo4j connection.

        Args:
            env_file (str): Path to the .env file containing configuration settings for the Neo4j connection and models.
            index_name (str): The name of the index in the Neo4j database to be used for querying.
        
        Raises:
            KeyError: If required environment variables are missing.
        """
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.platform = platform.system()

        # Neo4j connection parameters
        self.neo4j_url = os.environ["NEO4J_URI"].replace("http", "bolt")
        self.neo4j_username = os.environ["NEO4J_USERNAME"]
        self.neo4j_password = os.environ["NEO4J_PASSWORD"]

        self._start_server()

        # Model configuration
        self.model_name = os.environ["MODEL_LLM_NEO4J"]
        self.modelGroq_name = os.environ["GROQ_MODEL_NAME"]
        self.embedding_model = OllamaEmbeddings(model=self.model_name)
        self.llm_model = ChatGroq(model=self.modelGroq_name)
        self.index_name = index_name

    def __del__(self):
        self._stop_server()

    def _start_server(self):
        """
        Starts the Ollama server as a separate background process.
        """
        self.logger.info("Starting Ollama server...")
        try:
            if self.platform == "Darwin":
                try:
                    # Start the Ollama server in a separate process
                    self.process = subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.logger.info("Ollama server started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'ollama' command not found. Ensure Ollama is installed and in PATH.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting the server: {e}")
            elif self.platform == "Windows":
                try:
                    # Start the Ollama server in a separate process
                    powershell_command = ('Start-Process "cmd" -ArgumentList "/c ollama serve" -Verb runAs -WindowStyle Hidden')

                    self.process = subprocess.Popen(["powershell", "-Command", powershell_command])

                    self.logger.info("Ollama server started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'ollama' command not found. Ensure Ollama is installed and in PATH.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting the server: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while starting Neo4j console with your platform, make sure you are on Windows or macOS: {e}")


    def _stop_server(self):
        """
        Stops the Ollama server if it is running.
        """
        def is_process_running(pid):
            """Check if a process with a given PID is still running."""
            return psutil.pid_exists(pid)
        
        def is_port_in_use(port):
            """Check if a specific port is in use (useful for verifying if the server is active)."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0
        
        if self.process and is_process_running(self.process.pid):  # Check if the process is still active
            self.logger.info("Stopping the Ollama server...")
            self.process.terminate()  # Send a terminate signal
            self.process.wait()  # Wait for the process to terminate
            
            # Double-check if the process is still running
            if is_process_running(self.process.pid) or is_port_in_use(11434):  # Assuming Ollama runs on port 11434
                self.logger.warning("Failed to stop the Ollama server. Forcing termination...")
                self.process.kill()  # Force kill the process
            else:
                self.logger.info("Ollama server stopped successfully.")
        else:
            self.logger.warning("Ollama server is not running.")

    def query_similarity(self, query):
        """
        Creates a vector index on the Neo4j graph and performs a similarity-based query on the vector index.

        Args:
            query (str): The query string to be executed for similarity-based retrieval from the Neo4j graph.
        
        Raises:
            Exception: If there is an error during the execution of the similarity query.
        
        Returns:
            str: The result of the similarity query, or a message indicating no results were found.
        """
        
        node_label = "Article"
        text_node_properties = ["topic", "title", "body"]
        embedding_node_property = "embedding"
        
        retriever = Neo4jVector.from_existing_graph(
            self.embedding_model,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property
        ).as_retriever()
        
        vector_qa = RetrievalQA.from_chain_type(
            llm=self.llm_model, chain_type="stuff", retriever=retriever
        )
        
        self.logger.info(f"Executing similarity query...")
        try:
            start_time_similarity = time.time()
            result = vector_qa.invoke({"query": query})
            elapsed_time = time.time() - start_time_similarity
            self.logger.info(f"Similarity query completed in {elapsed_time:.2f} seconds.")
            return result.get("result", "No results found.")
        except Exception as e:
            self.logger.error(f"Error during similarity query: {e}")
            return None
        