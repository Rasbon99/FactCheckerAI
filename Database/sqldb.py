import os
import sqlite3
import glob
import dotenv
import os
import shutil

from log import Logger

class Database:
    def __init__(self, env_file="key.env"):
        """
        Initializes the Database class with the database file path.

        Args:
            env_file (str): The env file containing the API keys or database file paths. Default: "key.env".
        
        Raises:
            KeyError: If the environment variable for the database file path is not set.
        """
        
        self.logger = Logger(self.__class__.__name__).get_logger()
        try:
            dotenv.load_dotenv(env_file, override=True)
            self.db_file = os.environ["SQLDB_PATH"]
            self.assets_dir = os.environ["ASSET_PATH"]
        except KeyError as e:
            self.logger.error("Environment variable SQLDB_PATH not found.")
            raise e

        db_dir = os.path.dirname(self.db_file)  
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            self.logger.info("Created directory for the database file: %s", db_dir)

    def __enter__(self):
        """
        Establishes a connection to the SQLite database.

        Returns:
            sqlite3.Connection: A connection object to interact with the database.
        
        Raises:
            sqlite3.DatabaseError: If the connection to the database fails.
        """
        self.logger.info("Connecting to the SQLite database.")
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.conn.row_factory = sqlite3.Row
            self.logger.info("Successfully connected to the database.")
            return self.conn
        except sqlite3.DatabaseError as e:
            self.logger.error("Failed to connect to the database.")
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the SQLite database connection when the context manager is exited.

        Args:
            exc_type (type): The exception type, if any.
            exc_val (Exception): The exception value, if any.
            exc_tb (traceback): The traceback object, if any.
        
        Raises:
            Exception: If there is an error during the exit process of the database connection.
        """
        if self.conn:
            self.logger.info("Closing the database connection.")
            try:
                self.conn.close()
            except Exception as e:
                self.logger.error("Error while closing the database connection.")
                raise e

    def create_table(self, create_table_sql):
        """
        Creates a table in the SQLite database using the provided SQL statement.

        Args:
            create_table_sql (str): The SQL statement used to create the table.
        
        Raises:
            sqlite3.DatabaseError: If there is an error while creating the table.
        """
        self.logger.info("Creating table with SQL...")
        try:
            with self as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                conn.commit()
            self.logger.info("Table created successfully.")
        except sqlite3.DatabaseError as e:
            self.logger.error("Error creating table.")
            raise e

    def execute_query(self, query, params=()):
        """
        Executes a query on the SQLite database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): The parameters to pass with the query. Default is an empty tuple.
        
        Raises:
            sqlite3.DatabaseError: If there is an error during query execution.
        """

        masked_params = [param if not isinstance(param, (bytes, bytearray)) else "BLOB" for param in params]
        self.logger.info("Executing query: %s", query)

        try:
            with self as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            self.logger.info("Query executed successfully.")
        except sqlite3.DatabaseError as e:
            self.logger.error("Error executing query.")
            raise e

    def fetch_all(self, query, params=()):
        """
        Fetches all records that match the provided SQL query.

        Args:
            query (str): The SQL query to fetch records.
            params (tuple): The parameters to pass with the query. Default is an empty tuple.

        Returns:
            list: A list of rows returned by the query.
        
        Raises:
            sqlite3.DatabaseError: If there is an error during the fetch operation.
        """
        self.logger.info("Fetching all records for query: %s with params: %s", query, params)
        try:
            with self as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchall()
            self.logger.info("Fetched %d records.", len(result))
            return result
        except sqlite3.DatabaseError as e:
            self.logger.error("Error fetching records.")
            raise e
    
    def fetch_one(self, query, params=()):
        """
        Fetches the first record that matches the provided SQL query.

        Args:
            query (str): The SQL query to fetch records.
            params (tuple): The parameters to pass with the query. Default is an empty tuple.

        Returns:
            sqlite3.Row: The first row returned by the query.
        
        Raises:
            sqlite3.DatabaseError: If there is an error during the fetch operation.
        """
        self.logger.info("Fetching one record for query: %s with params: %s", query, params)
        try:
            with self as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
            self.logger.info("Fetched one record.")
            return result
        except sqlite3.DatabaseError as e:
            self.logger.error("Error fetching record.")
            raise e

    def delete_all_conversations(self):
        """
        Deletes data from tables "claims", "answers", and "sources", and cleans up all images in subdirectories of the assets folder.

        Args:
            None

        Raises:
            sqlite3.DatabaseError: If there is an error during the deletion process.
            OSError: If there is an error during the image deletion process.

        Returns:
            None
        """
        self.logger.info("Deleting all conversations and cleaning up subdirectories in assets.")
        try:
            # Deleting conversations from the database
            with self as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM claims;")
                cursor.execute("DELETE FROM answers;")
                cursor.execute("DELETE FROM sources;")
                conn.commit()
            self.logger.info("All conversations deleted successfully.")

            # Clean up subdirectories in the assets folder
            if os.path.isdir(self.assets_dir):
                for root, dirs, _ in os.walk(self.assets_dir, topdown=False):
                    # Remove all subdirectories
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        try:
                            shutil.rmtree(dir_path)  # Recursively delete the directory and its contents
                            self.logger.info(f"Deleted directory and its contents: {dir_path}")
                        except OSError as e:
                            self.logger.error(f"Error deleting directory {dir_path}: {e}")
            else:
                self.logger.warning(f"Assets folder does not exist: {self.assets_dir}")
                
        except sqlite3.DatabaseError as e:
            self.logger.error("Error deleting conversations.")
            raise e
        except OSError as e:
            self.logger.error("Error cleaning up assets.")
            raise e
        
    def get_history(self):
        """
        Retrieves the conversations from the database, including associated sources.

        Returns:
            pd.DataFrame: A DataFrame containing the conversations and their sources.
        """

        # Query per ottenere le conversazioni con risposta e immagine
        query = """
        SELECT c.id, c.text, c.title, a.answer, a.graphs_folder 
        FROM claims c
        INNER JOIN answers a ON c.id = a.claim_id
        """

        rows = self.fetch_all(query)

        if not rows:
            return {}

        conversations = []

        for row in rows:
            claim_id = row[0]
            
            # Query per ottenere le sources associate al claim
            sources_query = """
            SELECT title, url, body 
            FROM sources 
            WHERE claim_id = ?
            """
            sources_rows = self.fetch_all(sources_query, (claim_id,))

            # Formattare le sources come una lista di dizionari
            sources = [{"title": s[0], "url": s[1], "body": s[2]} for s in sources_rows]

            images = []
            graphs_folder = row[4]
            if graphs_folder and os.path.isdir(graphs_folder):
                jpg_files = glob.glob(os.path.join(graphs_folder, "*.jpg"))
            else:
                self.logger.warning("La cartella dei grafici non esiste o non è stata specificata.")

            conversations.append({
                "id": claim_id,
                "claim": row[1],
                "title": row[2],
                "answer": row[3],
                "images": jpg_files,
                "sources": sources 
            })

        return conversations