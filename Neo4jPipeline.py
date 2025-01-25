import dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
import time
from log import Logger

class RAG_Pipeline:
    def __init__(self, env_file="key.env", index_name="articles"):
        # Load environment variables
        dotenv.load_dotenv(env_file, override=True)
        
        self.logger = Logger(self.__class__.__name__).get_logger()
        
        self.model_name = os.environ["MODEL_LLM_NEO4J"]

        # Neo4j connection parameters
        self.neo4j_url = os.environ["NEO4J_URI"].replace("http", "bolt")
        self.neo4j_username = os.environ["NEO4J_USERNAME"]
        self.neo4j_password = os.environ["NEO4J_PASSWORD"]

        # Initialize graph connection
        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )
        
        try:
            self.graph.query("RETURN 1")
            self.logger.info("Connessione a Neo4j stabilita con successo.")
        except Exception as e:
            self.logger.error(f"Errore durante la connessione a Neo4j: {e}")
            raise ConnectionError(f"Errore durante la connessione a Neo4j: {e}")

        # Initialize embedding and LLM models
        self.embedding_model = OllamaEmbeddings(model=self.model_name)
        self.llm_model = ChatOllama(model=self.model_name)
        self.index_name = index_name

    def _load_data(self, csv_path):
        """
        Carica i dati da un file CSV nel grafo Neo4j.

        Args:
            csv_path (str): Path del file CSV contenente gli articoli da caricare.
        
        Raises:
            Exception: Se c'è un errore durante il caricamento dei dati.
        """

        q_load_articles = f"""
        LOAD CSV WITH HEADERS
        FROM '{csv_path}'
        AS row 
        FIELDTERMINATOR ';'
        MERGE (a:Article {{title:row.Title}})
        SET a.abstract = row.Abstract,
            a.publication_date = date(row.Publication_Date)
        FOREACH (researcher in split(row.Authors, ',') | 
            MERGE (p:Researcher {{name:trim(researcher)}})
            MERGE (p)-[:PUBLISHED]->(a))
        FOREACH (topic in [row.Topic] | 
            MERGE (t:Topic {{name:trim(topic)}})
            MERGE (a)-[:IN_TOPIC]->(t))
        """
        
        self.logger.info(f"Avvio caricamento dati da {csv_path}...")
        try:
            start_time = time.time()
            self.graph.query(q_load_articles)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Caricamento completato in {elapsed_time:.2f} secondi.")
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento dati: {e}")
        
        self.graph.refresh_schema()

    def _query_similarity(self, query):
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
            result = vector_qa.invoke({"query": query})
            self.logger.info("Query di similarità completata.")
            return result.get("result", "Nessun risultato trovato.")
        except Exception as e:
            self.logger.error(f"Errore durante la query di similarità: {e}")
            return None  
        
    def run_pipeline(self, csv_path, query):
            """
            Esegui l'intera pipeline: carica i dati dal CSV e poi esegui la query di similarità.

            Args:
                csv_path (str): Percorso del file CSV.
                query (str): La query di similarità da eseguire.
            
            Returns:
                str: Risultato della query di similarità.
            """
            
            self._load_data(csv_path)
            return self._query_similarity(query)

# Usage example
if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = RAG_Pipeline()
    
    csv_path = "https://raw.githubusercontent.com/dcarpintero/generative-ai-101/main/dataset/synthetic_articles.csv"
    query = "which articles discuss how AI might affect our daily life? include the article titles and abstracts."
    
    query_result = pipeline.run_pipeline(
        csv_path=csv_path,
        query=query
    )
    print(query_result)