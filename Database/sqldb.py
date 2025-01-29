import os
import sys
import sqlite3

import dotenv

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

dotenv.load_dotenv("key.env", override=True)

class Database:
    def __init__(self, env_file="Pkey.env", db_file=os.getenv("SQLDB_PATH")):
        """
        Initializes the Database class with the database file path.

        Args:
            env_file (str): The env file containing the API keys or database file paths. Default: "key.env".
            db_file (str): The path to the SQLite database file.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.db_file = db_file

        db_dir = os.path.dirname(db_file)  
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            self.logger.info("Created directory for the database file: %s", db_dir)

    def __enter__(self):
        """
        Establishes a connection to the SQLite database.

        Returns:
            sqlite3.Connection: A connection object to interact with the database.
        """
        self.logger.info("Connecting to the SQLite database.")
        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row
        self.logger.info("Successfully connected to the database.")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the SQLite database connection when the context manager is exited.

        Args:
            exc_type (type): The exception type, if any.
            exc_val (Exception): The exception value, if any.
            exc_tb (traceback): The traceback object, if any.
        """
        if self.conn:
            self.logger.info("Closing the database connection.")
            self.conn.close()

    def create_table(self, create_table_sql):
        """
        Creates a table in the SQLite database using the provided SQL statement.

        Args:
            create_table_sql (str): The SQL statement used to create the table.
        """
        self.logger.info("Creating table with SQL: %s", create_table_sql)
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
        self.logger.info("Table created successfully.")

    def execute_query(self, query, params=()):
        """
        Executes a query on the SQLite database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): The parameters to pass with the query. Default is an empty tuple.
        """
        self.logger.info("Executing query: %s with params: %s", query, params)
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
        self.logger.info("Query executed successfully.")

    def fetch_all(self, query, params=()):
        """
        Fetches all records that match the provided SQL query.

        Args:
            query (str): The SQL query to fetch records.
            params (tuple): The parameters to pass with the query. Default is an empty tuple.

        Returns:
            list: A list of rows returned by the query.
        """
        self.logger.info("Fetching all records for query: %s with params: %s", query, params)
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
        self.logger.info("Fetched %d records.", len(result))
        return result
