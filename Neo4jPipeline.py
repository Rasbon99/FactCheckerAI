import dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
import time
from log import Logger
from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        
        # TITLE, URL, SITE, BODY, ENTITY
        
        #Title;URL;Site;Body;Entities;
        #"OpenAI lancia GPT-4";"https://example.com/openai-gpt4";"TechCrunch";"OpenAI, GPT-4, Intelligenza Artificiale"
        
        self.logger.info(f"Avvio caricamento dati da {csv_path}...")
        try:
            start_time = time.time()
            self.graph.query(q_load_articles)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Caricamento completato in {elapsed_time:.2f} secondi.")
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento dati: {e}")
        
        self.graph.refresh_schema()

    def _extract_and_save_graph(self,  output_file_topic="graph_topic.jpg", output_file_published="graph_published.jpg"):
        """
        Esegue una query su Neo4j, crea il grafo e lo salva su file JPEG.
        """ 
        try:
            
            blue_light = "#add8e6"  # Blu chiaro
            green_light = "#90ee90"  # Verde chiaro

            graph = Graph(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password))
            
            # Primo grafico: (Article)-[:IN_TOPIC]->(Topic)
            query_topic = """
            MATCH (a:Article)-[:IN_TOPIC]->(t:Topic)
            RETURN a.title AS Articolo,
                   t.name AS Topic
            """
            results_topic = graph.run(query_topic).to_data_frame()
            
            G_topic = nx.DiGraph()
            for _, row in results_topic.iterrows():
                G_topic.add_edge(row['Articolo'], row['Topic'])

            # Generare una lista di colori per i topic
            colors = list(mcolors.TABLEAU_COLORS.values())  # Puoi scegliere anche altri set di colori
            color_map = {}

            # Assegna un colore unico a ogni topic
            unique_topics = results_topic['Topic'].unique()
            for i, topic in enumerate(unique_topics):
                color_map[topic] = colors[i % len(colors)]  # Ricicla i colori se i topic sono più dei colori disponibili

            # Colori per i nodi
            node_colors = []
            for node in G_topic.nodes():
                if node in color_map:  # Se è un topic
                    node_colors.append(color_map[node])
                else:  # Se è un articolo, assegna un colore neutro (ad esempio grigio)
                    node_colors.append(blue_light)

            # Colori per le frecce (edge)
            edge_colors = []
            for u, v in G_topic.edges():
                edge_colors.append(color_map[v])  # Colore della freccia basato sul topic (nodo di destinazione)

            # Abbreviazione etichette se troppo lunghe
            max_len = 15
            labels_top = {}
            for node in G_topic.nodes():
                label_text = node if len(node) <= max_len else node[:max_len] + "..."
                labels_top[node] = label_text

            # Layout del grafico
            pos_topic = nx.spring_layout(G_topic, k=2, iterations=50)

            # Disegna il grafico
            plt.figure(figsize=(12, 9))
            nx.draw(
                G_topic,
                labels=labels_top,
                pos=pos_topic,
                with_labels=True,
                node_color=node_colors,  # Assegna i colori ai nodi
                edge_color=edge_colors,  # Assegna i colori alle frecce
                node_size=3500,
                font_size=10,
                width=2  # Aumenta la larghezza delle frecce per migliorarne la visibilità
            )
            plt.savefig(output_file_topic, dpi=500)

            # Secondo grafico: (Ricercatore)-[:PUBLISHED]->(Article)
            query_published = """
            MATCH (p:Researcher)-[:PUBLISHED]->(a:Article)
            RETURN p.name AS Ricercatore,
                   a.title AS Articolo
            """
            results_pub = graph.run(query_published).to_data_frame()
            G_pub = nx.DiGraph()
            for _, row in results_pub.iterrows():
                G_pub.add_edge(row['Ricercatore'], row['Articolo'])

            # Generare una lista di colori
            colors = list(mcolors.TABLEAU_COLORS.values())  # Puoi scegliere anche altri set di colori
            color_map = {}

            # Assegna un colore unico a ogni ricercatore
            unique_researchers = results_pub['Ricercatore'].unique()
            for i, researcher in enumerate(unique_researchers):
                color_map[researcher] = colors[i % len(colors)]  # Ricicla i colori se i ricercatori sono più dei colori disponibili

            # Colori per i nodi
            node_colors = []
            for node in G_pub.nodes():
                if node in color_map:  # Se è un ricercatore
                    node_colors.append(color_map[node])
                else:  # Se è un articolo, assegna un colore neutro (ad esempio grigio)
                    node_colors.append(green_light)

            # Colori per le frecce (edge)
            edge_colors = []
            for u, v in G_pub.edges():
                edge_colors.append(color_map[u])  # Colore della freccia basato sul ricercatore (nodo di partenza)

            # Abbreviazione etichette se troppo lunghe
            max_len = 10
            labels_pub = {}
            for node in G_pub.nodes():
                label_text = node if len(node) <= max_len else node[:max_len] + "..."
                labels_pub[node] = label_text

            # Layout del grafico
            pos_pub = nx.spring_layout(G_pub, k=2, iterations=50)

            # Disegna il grafico
            plt.figure(figsize=(12, 9))
            nx.draw(
                G_pub,
                labels=labels_pub,
                pos=pos_pub,
                with_labels=True,
                node_color=node_colors,  # Assegna i colori ai nodi
                edge_color=edge_colors,  # Assegna i colori alle frecce
                node_size=3500,
                font_size=10,
                width=2  # Aumenta la larghezza delle frecce per migliorarne la visibilità
            )
            plt.savefig(output_file_published, dpi=500)
            plt.show()

        except Exception as e:
            self.logger.error(f"Errore durante l'estrazione e il salvataggio del grafo: {e}")

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
            start_time_similarity = time.time()
            result = vector_qa.invoke({"query": query})
            elapsed_time = time.time() - start_time_similarity
            self.logger.info(f"Query di similarità completata in {elapsed_time:.2f} secondi.")
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
            self._extract_and_save_graph()
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