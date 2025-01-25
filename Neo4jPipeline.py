import dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

class Neo4jPipeline:
    def __init__(self, env_file="key.env", model_name="phi3.5:latest", index_name="articles"):
        # Load environment variables
        dotenv.load_dotenv(env_file, override=True)

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

        # Initialize embedding and LLM models
        self.embedding_model = OllamaEmbeddings(model=model_name)
        self.llm_model = ChatOllama(model=model_name)
        self.index_name = index_name

    def load_data(self, csv_url):
        """Load data into the Neo4j graph."""
        q_load_articles = f"""
        LOAD CSV WITH HEADERS
        FROM '{csv_url}'
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
        self.graph.query(q_load_articles)
        self.graph.refresh_schema()

    def create_vector_index(self, node_label, text_node_properties, embedding_node_property):
        """Create a vector index on the Neo4j graph."""
        self.vector_index = Neo4jVector.from_existing_graph(
            self.embedding_model,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
        )

    def query_similarity(self, query):
        """Perform a similarity-based query on the vector index."""
        retriever = self.vector_index.as_retriever()
        vector_qa = RetrievalQA.from_chain_type(
            llm=self.llm_model, chain_type="stuff", retriever=retriever
        )
        result = vector_qa.invoke({"query": query})
        return result["result"]

# Usage example
if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = Neo4jPipeline()

    pipeline.load_data(
        csv_url="https://raw.githubusercontent.com/dcarpintero/generative-ai-101/main/dataset/synthetic_articles.csv"
    )

    # Create vector index
    pipeline.create_vector_index(
        node_label="Article",
        text_node_properties=["topic", "title", "abstract"],
        embedding_node_property="embedding",
    )

    # Perform a similarity query
    query_result = pipeline.query_similarity(
        "which articles discuss how AI might affect our daily life? include the article titles and abstracts."
    )
    print(query_result)
