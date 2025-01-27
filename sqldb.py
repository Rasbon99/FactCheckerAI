import sqlite3
import dotenv
import os

dotenv.load_dotenv("key.env", override=True)

class Database:
    def __init__(self, env_file="key.env",db_file= os.getenv("SQLDB_PATH")):
        
        self.db_file = db_file

        db_dir = os.path.dirname(db_file)  # Ottieni la directory dal percorso
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)  # Crea la directory se non esiste

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def create_table(self, create_table_sql):
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()

    def execute_query(self, query, params=()):
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

    def fetch_all(self, query, params=()):
        with self as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()