import os
import time
import platform

import dotenv
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from llamacpp_client import ChatLlamaCppServer

from log import Logger


class TokenTrackerCallback(BaseCallbackHandler):
    """Listens to LangChain LLM calls to extract token usage and call counts."""

    def __init__(self):
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        # Safely extract token usage whether it's stored in llm_output or message metadata
        if response.llm_output and "token_usage" in response.llm_output:
            self.total_tokens += response.llm_output["token_usage"].get(
                "total_tokens", 0
            )
        elif response.generations and len(response.generations) > 0:
            first_gen = response.generations[0][0]
            if hasattr(first_gen, "message") and hasattr(
                first_gen.message, "response_metadata"
            ):
                self.total_tokens += first_gen.message.response_metadata.get(
                    "token_usage", {}
                ).get("total_tokens", 0)


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

        # Model configuration names from environment
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"
        )
        self.model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")

        # Initialize Hugging Face embeddings natively in Python memory
        self.logger.info(f"Loading local embedding model: {self.embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm_model = ChatLlamaCppServer(
            model=self.model_alias,
            temperature=0.1,
            max_tokens=2048,
        )
        self.index_name = index_name

    def query_similarity(self, query):
        """
        Performs a vector-based similarity search and RAG query on the Neo4j graph.

        Args:
            query (str): The claim and instructions to be processed by the LLM.

        Returns:
            tuple: A tuple containing:
                - str: The verdict/answer from the LLM (e.g., Supported, Refuted, NEI).
                - dict: Token usage metadata for RQ2 Efficiency (total tokens and call count).

        Raises:
            None: Exceptions are caught internally and logged to ensure the pipeline continues.
        """
        self.logger.info("Executing similarity query...")
        token_tracker = TokenTrackerCallback()
        vector_store = None

        try:
            start_time_similarity = time.time()

            # 1. Initialize here so it calculates embeddings for the NEW evidence
            self.logger.info("Syncing embeddings and initializing Retriever...")
            vector_store = Neo4jVector.from_existing_graph(
                self.embedding_model,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                node_label="Article",
                text_node_properties=["topic", "title", "body"],
                embedding_node_property="embedding",
            )

            retriever = vector_store.as_retriever()
            vector_qa = RetrievalQA.from_chain_type(
                llm=self.llm_model, chain_type="stuff", retriever=retriever
            )

            result = vector_qa.invoke(
                {"query": query}, config={"callbacks": [token_tracker]}
            )

            elapsed_time = time.time() - start_time_similarity
            self.logger.info(
                f"Similarity query completed in {elapsed_time:.2f} seconds."
            )

            # Package the extracted metrics
            token_data = {
                "total": token_tracker.total_tokens,
                "calls": token_tracker.llm_calls,
            }
            return result.get("result", "No results found."), token_data

        except Exception as e:
            self.logger.error(f"Error during similarity query: {e}")
            return None, {"total": 0, "calls": 0}

        finally:
            # 2. STRICT CLEANUP: Close the connection pool to prevent the RAM freeze!
            if vector_store is not None:
                try:
                    # LangChain's Neo4jVector stores the active driver here
                    vector_store._driver.close()
                    self.logger.info("Neo4j vector store connection closed safely.")
                except Exception as e:
                    self.logger.warning(f"Could not close Neo4j connection: {e}")
