import os
import sys
import sqlite3

import dotenv

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

class Database:
    def __init__(self, env_file="Dkey.env"):
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
        self.logger.info("Creating table with SQL: %s", create_table_sql)
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
        self.logger.info("Executing query: %s with params: %s", query, params)
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