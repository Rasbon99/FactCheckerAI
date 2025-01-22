import dotenv
import os
from langchain_neo4j import Neo4jGraph  

dotenv.load_dotenv("key.env", override=True)

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"].replace("http", "bolt"),  # Corregge lo schema URI
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)

q_load_articles = """
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/dcarpintero/generative-ai-101/main/dataset/synthetic_articles.csv' 
AS row 
FIELDTERMINATOR ';'
MERGE (a:Article {title:row.Title})
SET a.abstract = row.Abstract,
    a.publication_date = date(row.Publication_Date)
FOREACH (researcher in split(row.Authors, ',') | 
    MERGE (p:Researcher {name:trim(researcher)})
    MERGE (p)-[:PUBLISHED]->(a))
FOREACH (topic in [row.Topic] | 
    MERGE (t:Topic {name:trim(topic)})
    MERGE (a)-[:IN_TOPIC]->(t))
"""

graph.query(q_load_articles)
graph.refresh_schema()
print(graph.get_schema)


# VECTOR INDEX
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="phi3.5:latest")

vector_index = Neo4jVector.from_existing_graph(
    embedding_model,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="articles",
    node_label="Article",
    text_node_properties=["topic", "title", "abstract"],
    embedding_node_property="embedding",
)

#Q&A ON SIMILARITY
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

llm_model = ChatOllama(model="phi3.5:latest")

vector_qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", retriever=vector_index.as_retriever())

r = vector_qa.invoke(
    {
        "query": "which articles discuss how AI might affect our daily life? include the article titles and abstracts."
    }
)
print(r["result"])