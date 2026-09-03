import os
import time
import platform

import dotenv
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from Utils.nomic_embedding import get_embedding_model


from log import Logger


class TokenTrackerCallback(BaseCallbackHandler):
    """Listens to LangChain LLM calls to extract token usage and call counts."""

    def __init__(self):
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        # ChatGroq populates token usage inside the llm_output dictionary
        if response.llm_output and "token_usage" in response.llm_output:
            self.total_tokens += response.llm_output["token_usage"].get(
                "total_tokens", 0
            )


class QueryEngine:

    def __init__(self, env_file="key.env", index_name="articles_nomic"):
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
        self.modelGroq_name = os.environ["GROQ_MODEL_NAME"]

        # Initialize Hugging Face embeddings natively in Python memory
        self.logger.info(f"Loading local embedding model: {self.embedding_model_name}")
        self.embedding_model = get_embedding_model(self.embedding_model_name)

        self.llm_model = ChatGroq(model=self.modelGroq_name)
        self.index_name = index_name

    def query_similarity(self, query):
        """
        Performs a hybrid GraphRAG query: Vector search + Graph Traversal on the Neo4j graph.

        Args:
            query (str): The claim and instructions to be processed by the LLM.

        Returns:
            tuple: A tuple containing:
                - str: The verdict/answer from the LLM (e.g., Supported, Refuted, NEI).
                - dict: Token usage metadata for RQ2 Efficiency (total tokens and call count).
        """
        self.logger.info("Executing Hybrid GraphRAG query...")
        token_tracker = TokenTrackerCallback()
        vector_store = None

        try:
            start_time_similarity = time.time()

            # --- TRUE GRAPHRAG IMPLEMENTATION ---
            # This Cypher query executes AFTER the vector search finds the most similar 'node'.
            # It traverses the graph edges to grab the associated Topic, Site, and Entities,
            # and returns them as a single, context-rich 'text' block for the LLM.
            graph_retrieval_query = """
            OPTIONAL MATCH (node)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (node)-[:PUBLISHED_ON]->(s:Site)
            OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
            WITH node, score, t, s, collect(DISTINCT e.name) AS entities
            
            WITH node, score, t, s, entities,
                 reduce(acc = "", ent IN entities | acc + ent + ", ") AS entities_str
                 
            RETURN "Title: " + coalesce(node.title, "Unknown") + "\\n" +
                   "Topic: " + coalesce(t.name, "Unknown") + "\\n" +
                   "Source: " + coalesce(s.name, "Unknown") + "\\n" +
                   "Entities: " + coalesce(entities_str, "None") + "\\n" +
                   "Content: " + coalesce(node.body, "") AS text,
                   score,
                   {title: node.title} AS metadata
            """

            self.logger.info("Syncing embeddings and initializing Graph Retriever...")

            vector_store = Neo4jVector.from_existing_graph(
                self.embedding_model,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                node_label="Article",
                text_node_properties=[
                    "title",
                    "body",
                ],  # 'topic' is now fetched via the graph!
                embedding_node_property="embedding",
                retrieval_query=graph_retrieval_query,  # Injecting the graph traversal here!
            )

            retriever = vector_store.as_retriever()
            vector_qa = RetrievalQA.from_chain_type(
                llm=self.llm_model, chain_type="stuff", retriever=retriever
            )

            result = vector_qa.invoke(
                {"query": query}, config={"callbacks": [token_tracker]}
            )

            elapsed_time = time.time() - start_time_similarity
            self.logger.info(f"GraphRAG query completed in {elapsed_time:.2f} seconds.")

            # Package the extracted metrics
            token_data = {
                "total": token_tracker.total_tokens,
                "calls": token_tracker.llm_calls,
            }
            return result.get("result", "No results found."), token_data

        except Exception as e:
            self.logger.error(f"Error during GraphRAG query: {e}")
            return None, {"total": 0, "calls": 0}

        finally:
            # 2. STRICT CLEANUP: Close the connection pool to prevent the RAM freeze!
            if vector_store is not None:
                try:
                    vector_store._driver.close()
                    self.logger.info("Neo4j vector store connection closed safely.")
                except Exception as e:
                    self.logger.warning(f"Could not close Neo4j connection: {e}")
