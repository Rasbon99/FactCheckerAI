import dotenv
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import time
import sys

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

class QueryEngine:
    def __init__(self, env_file="RAGkey.env", index_name="articles"):
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Neo4j connection parameters
        self.neo4j_url = os.environ["NEO4J_URI"].replace("http", "bolt")
        self.neo4j_username = os.environ["NEO4J_USERNAME"]
        self.neo4j_password = os.environ["NEO4J_PASSWORD"]

        # Model configuration
        self.model_name = os.environ["MODEL_LLM_NEO4J"]
        self.modelGroq_name = os.environ["MODELGROQ_LLM_NEO4J"]
        self.embedding_model = OllamaEmbeddings(model=self.model_name)
        self.llm_model = ChatGroq(model=self.modelGroq_name)
        self.index_name = index_name

    def query_similarity(self, query):
        """Create a vector index on the Neo4j graph and perform a similarity-based query on the vector index."""
        
        node_label="Article"
        text_node_properties=["topic", "title", "abstract"]
        embedding_node_property="embedding"
        
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
                    
        self.logger.info(f"Esecuzione query di similarità: {query}")
        try:
            start_time_similarity = time.time()
            result = vector_qa.invoke({"query": query})
            elapsed_time = time.time() - start_time_similarity
            self.logger.info(f"Query di similarità completata in {elapsed_time:.2f} secondi.")
            return result.get("result", "Nessun risultato trovato.")
        except Exception as e:
            self.logger.error(f"Errore durante la query di similarità: {e}")
            return None  