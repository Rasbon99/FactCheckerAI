import os
import time
import platform

import dotenv
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from log import Logger


class TokenTrackerCallback(BaseCallbackHandler):
    """Listens to LangChain LLM calls to extract token usage and call counts."""

    def __init__(self):
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1

        # Fallback 1: Standard llm_output location (Older LangChain versions)
        if response.llm_output and "token_usage" in response.llm_output:
            self.total_tokens += response.llm_output["token_usage"].get(
                "total_tokens", 0
            )
            return

        # Fallback 2: Message metadata location (Newer LangChain versions)
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    if (
                        hasattr(gen, "message")
                        and hasattr(gen.message, "usage_metadata")
                        and gen.message.usage_metadata
                    ):
                        self.total_tokens += gen.message.usage_metadata.get(
                            "total_tokens", 0
                        )
        except Exception:
            pass


class QueryEngine:
    def __init__(self, env_file="key.env", index_name="articles"):
        """
        Initializes the QueryEngine by setting up the environment variables, models, and Neo4j connection.

        Args:
            env_file (str): Path to the .env file containing configuration settings.
            index_name (str): The name of the index in the Neo4j database to be used for querying.
        """
        dotenv.load_dotenv(env_file, override=False)
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.platform = platform.system()

        self.neo4j_url = os.environ["NEO4J_URI"].replace("http", "bolt")
        self.neo4j_username = os.environ["NEO4J_USERNAME"]
        self.neo4j_password = os.environ["NEO4J_PASSWORD"]

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"
        )
        self.modelGroq_name = os.environ["GROQ_MODEL_NAME"]

        self.logger.info(f"Loading local embedding model: {self.embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm_model = ChatGroq(model=self.modelGroq_name)
        self.index_name = index_name

    def query_similarity(self, query):
        """
        Performs a vector-based similarity search and RAG query on the Neo4j graph.

        Args:
            query (str): The claim and instructions to be processed by the LLM.

        Returns:
            tuple: A tuple containing:
                - str: The verdict/answer from the LLM (e.g., Supported, Refuted, NEI).
                - dict: Token usage metadata (total tokens and call count).
        """
        self.logger.info("Executing similarity query...")
        token_tracker = TokenTrackerCallback()
        vector_store = None

        try:
            start_time_similarity = time.time()

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

            token_data = {
                "total": token_tracker.total_tokens,
                "calls": token_tracker.llm_calls,
            }
            return result.get("result", "No results found."), token_data

        except Exception as e:
            self.logger.error(f"Error during similarity query: {e}")
            return None, {"total": 0, "calls": 0}

        finally:
            if vector_store is not None:
                try:
                    vector_store._driver.close()
                    self.logger.info("Neo4j vector store connection closed safely.")
                except Exception as e:
                    self.logger.warning(f"Could not close Neo4j connection: {e}")
