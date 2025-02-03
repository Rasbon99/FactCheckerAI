import os
import time
import dotenv
import platform
import psutil
import socket
import subprocess

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from py2neo import Graph
from langchain_neo4j import Neo4jGraph

from log import Logger

class GraphManager:
    def __init__(self, env_file="key.env"):
        """
        Initializes the GraphManager by setting up the Neo4j connection.

        Args:
            env_file (str): Path to the .env file containing Neo4j credentials.
        
        Raises:
            ConnectionError: If there is an error during the connection to Neo4j.
        """
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.platform = platform.system()

        self._start_console()
        
        time.sleep(30)
        
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
            self.logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            self.logger.error(f"Error during Neo4j connection: {e}")
            raise ConnectionError(f"Error during Neo4j connection: {e}")
    
    def _start_console(self):
        """
        Starts the Neo4j console as a background process.
        """
        self.logger.info("Starting Neo4j console...")
        try:
            if self.platform == "Darwin":
                try:
                    # Avvia il comando "neo4j console" in un processo separato
                    self.process = subprocess.Popen(
                        ["neo4j", "console"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.logger.info("Neo4j console started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'neo4j' command not found.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting Neo4j console: {e}")
            elif self.platform == "Windows":
                try:
                    powershell_command = ('Start-Process "cmd" -ArgumentList "/c neo4j console" -Verb runAs -WindowStyle Hidden')

                    self.process = subprocess.Popen(["powershell", "-Command", powershell_command])

                    self.logger.info("Neo4j console started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'neo4j' command not found.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting Neo4j console: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while starting Neo4j console with your platform, make sure you are on Windows or macOS: {e}")

    def _stop_console(self):
        """
        Stops the Neo4j console if it is running.
        """
        def is_process_running(pid):
            """Check if a process with a given PID is still running."""
            return psutil.pid_exists(pid)
        
        def is_port_in_use(port):
            """Check if a specific port is in use (useful for verifying if the console is active)."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0
        
        if self.process and is_process_running(self.process.pid):  # Check if the process is still active
            self.logger.info("Stopping Neo4j console...")
            self.process.terminate()  # Send a terminate signal
            self.process.wait()  # Wait for the process to terminate
            
            # Double-check if the process is still running
            if is_process_running(self.process.pid) or is_port_in_use(7687):  # Assuming Neo4j runs on port 7687
                self.logger.warning("Failed to stop the Neo4j console. Forcing termination...")
                self.process.kill()  # Force kill the process
            else:
                self.logger.info("Neo4j console stopped successfully.")
        else:
            self.logger.warning("Neo4j console is not running.")

    def __del__(self):
        self._stop_console()
    
    def reset_data(self):
        """
        Resets the database by deleting all articles, topics, sites, and entities
        that are no longer connected to any articles.

        Logs the process and measures the execution time.
        """
        self.logger.info("Starting data reset process...")
        
        try:
            start_time = time.time()
            
            # Define the reset queries
            delete_queries = [
                "MATCH (a:Article) DETACH DELETE a",
                "MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DELETE e",
                "MATCH (s:Site) WHERE NOT (s)<-[:PUBLISHED_ON]-() DELETE s",
                "MATCH (t:Topic) WHERE NOT (t)<-[:HAS_TOPIC]-() DELETE t"
            ]
            
            # Execute each query
            for query in delete_queries:
                self.logger.info(f"Executing query: {query}")
                self.graph.query(query)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Data reset completed in {elapsed_time:.2f} seconds.")
            
            # Optional schema refresh
            self.graph.refresh_schema()
            self.logger.info("Schema refreshed successfully.")

        except Exception as e:
            self.logger.error(f"Error during data reset: {e}")

    def load_data(self, data):
        """
        Loads data into the Neo4j graph.

        Args:
            data (list): List of dictionaries containing news articles, including TITLE, URL, BODY, SITE, ENTITY, and TOPIC.
        
        Raises:
            Exception: If there is an error during data loading.
        
        Returns:
            None
        """
        q_load_articles = """
        UNWIND $data AS article
        MERGE (a:Article {title: article.title})
        SET a.url = article.url,
            a.body = article.body,
            a.score = article.score

        MERGE (s:Site {name: article.site})
        MERGE (a)-[:PUBLISHED_ON]->(s)

        // Gestione delle entità
        WITH a, article
        UNWIND article.entities AS entity
        MERGE (e:Entity {name: entity})
        MERGE (a)-[:MENTIONS]->(e)

        // Gestione del topic
        WITH a, article
        MERGE (t:Topic {name: article.topic})
        MERGE (a)-[:HAS_TOPIC]->(t)
        """
        
        self.logger.info(f"Starting data load from {data}...")
        try:
            start_time = time.time()
            self.graph.query(q_load_articles, params={"data": data})
            elapsed_time = time.time() - start_time
            self.logger.info(f"Loading completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            self.logger.error(f"Error during data loading: {e}")
        
        self.graph.refresh_schema()

    def extract_and_save_graph(self, output_file_topic, output_file_entity, output_file_site):
        """
        Executes a query on Neo4j, creates the graph, and saves it as a JPEG file.

        Args:
            output_file_topic (str): Path to save the topic graph.
            output_file_entity (str): Path to save the entity graph.
            output_file_site (str): Path to save the site graph.
        
        Raises:
            Exception: If there is an error during graph creation or saving.
        
        Returns:
            None
        """
        try:
            blue_light = "#add8e6"  

            graph = Graph(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password))
        
            def create_and_save_graph(query, node_relation, node_label, edge_label, output_file):
                """
                Creates and saves a graph based on a Cypher query and a relationship.

                Args:
                    query (str): The Cypher query to execute.
                    node_relation (tuple): A tuple containing the relation between the nodes (e.g., ('Article', 'Topic')).
                    node_label (str): Label for the nodes.
                    edge_label (str): Label for the edges.
                    output_file (str): Path to save the generated graph image.
                
                Raises:
                    Exception: If there is an error during graph creation or saving.
                
                Returns:
                    None
                """
                results = graph.run(query).to_data_frame()
                G = nx.DiGraph()
                
                # Add edges to the graph
                for _, row in results.iterrows():
                    G.add_edge(row[node_relation[0]], row[node_relation[1]], label=edge_label)

                # Generate a list of colors for the nodes
                colors = list(mcolors.TABLEAU_COLORS.values())
                color_map = {}
                unique_nodes = results[node_relation[1]].unique()  # Associate nodes with the second parameter of the relation
                for i, node in enumerate(unique_nodes):
                    color_map[node] = colors[i % len(colors)]  # Recycle colors if nodes exceed available colors

                # Node colors
                node_colors = []
                for node in G.nodes():
                    if node in color_map:
                        node_colors.append(color_map[node])
                    else:  # If it is an article, assign a neutral color
                        node_colors.append(blue_light)

                # Edge colors
                edge_colors = []
                edge_labels = {}  # To store edge labels
                for u, v, data in G.edges(data=True):
                    edge_colors.append(color_map[v])  # Arrow color based on the target node
                    edge_labels[(u, v)] = data['label']  # Edge label

                # Truncate labels if too long
                max_len = 15  # Maximum length for each line

                labels = {}

                def split_label(label, max_len):
                    """
                    Splits the label into two lines without truncating words.

                    Args:
                        label (str): The label to split.
                        max_len (int): The maximum length for each line.
                    
                    Raises:
                        None
                    
                    Returns:
                        str: The split label with two lines.
                    """
                    if len(label) <= max_len:
                        return label  # No split necessary
                    
                    # Split the first part without exceeding the length limit
                    first_line = label[:max_len]
                    
                    # Find the last space before the limit to avoid cutting off the word
                    if len(first_line) == max_len:
                        first_line = first_line[:first_line.rfind(' ')]  # Find the last space
                        second_line = label[len(first_line):]
                    else:
                        second_line = label[len(first_line):]
                    
                    # If the second part is too long, shorten it (only if necessary)
                    if len(second_line) > max_len:
                        second_line = second_line[:max_len] + "..."
                    
                    return f"{first_line}\n{second_line}"

                # Length parameters
                max_len = 15  # Maximum length for each line

                labels = {}

                for node in G.nodes():
                    node_label = f"{node}"  # Or any other text to associate with the node
                    
                    # Split the label
                    label_text = split_label(node_label, max_len)
                    
                    labels[node] = label_text

                # Graph layout
                pos = nx.kamada_kawai_layout(G)

                # Add a bit of "push" to avoid overlaps
                def avoid_overlap(pos, G, threshold=0.1):
                    """
                    Avoids overlap between nodes in the graph layout.

                    Args:
                        pos (dict): Dictionary containing node positions.
                        G (networkx.Graph): The graph object.
                        threshold (float): Minimum distance between nodes to avoid overlap.
                    
                    Raises:
                        None
                    
                    Returns:
                        dict: Updated positions of the nodes.
                    """
                    nodes = list(G.nodes())
                    overlap = True
                    while overlap:
                        overlap = False
                        for i, node_i in enumerate(nodes):
                            for j, node_j in enumerate(nodes):
                                if i >= j:
                                    continue
                                # Calculate the distance between two nodes
                                dist = np.linalg.norm(np.array(pos[node_i]) - np.array(pos[node_j]))
                                if dist < threshold:
                                    # If too close, push them apart
                                    pos[node_i] = [pos[node_i][0] + 0.1, pos[node_i][1] + 0.1]
                                    pos[node_j] = [pos[node_j][0] - 0.1, pos[node_j][1] - 0.1]
                                    overlap = True
                                    break
                    return pos

                # Apply the overlap avoidance function
                pos = avoid_overlap(pos, G)

                # Draw the graph
                plt.figure(figsize=(12, 9))
                nx.draw(
                    G,
                    labels=labels,
                    pos=pos,
                    with_labels=True,
                    node_color=node_colors,
                    edge_color=edge_colors,
                    node_size=3500,
                    font_size=6,
                    width=2
                )

                # Draw edge labels
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    edge_labels=edge_labels,
                    font_size=6,
                    font_color="black"
                )

                plt.savefig(output_file, dpi=500)

            # First graph: (Article)-[:HAS_TOPIC]->(Topic)
            query_topic = """
            MATCH (a:Article)-[:HAS_TOPIC]->(t:Topic)
            RETURN a.title AS Article,
                t.name AS Topic
            """
            create_and_save_graph(query_topic, node_relation=("Article", "Topic"), node_label="Topic", edge_label="HAS_TOPIC", output_file=output_file_topic)

            # Second graph: (Article)-[:MENTIONS]->(Entity)
            query_mentions = """
            MATCH (a:Article)-[:MENTIONS]->(e:Entity)
            RETURN a.title AS Article,
                e.name AS Entity
            """
            create_and_save_graph(query_mentions, node_relation=("Article", "Entity"), node_label="Entity", edge_label="MENTIONS", output_file=output_file_entity)

            # Third graph: (Article)-[:PUBLISHED_ON]->(Site)
            query_site = """
            MATCH (a:Article)-[:PUBLISHED_ON]->(s:Site)
            RETURN a.title AS Article,
                s.name AS Site
            """
            create_and_save_graph(query_site, node_relation=("Article", "Site"), node_label="Site", edge_label="PUBLISHED_ON", output_file=output_file_site)

            self.logger.info("Graphs generated and saved successfully.")
        except Exception as e:
            self.logger.error(f"Error during graph extraction and saving: {e}")