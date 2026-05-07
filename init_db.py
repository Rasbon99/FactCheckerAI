from Database.sqldb import Database
from log import Logger

logger = Logger(__name__).get_logger()


def initialize_database_schema(db: Database | None = None):
    """
    Creates all required SQLite tables if they do not already exist.

    Args:
        db (Database, optional): Existing Database instance. If omitted, a new instance is created.

    Returns:
        Database: The Database instance used for initialization.
    """
    db_manager = db if db is not None else Database()

    db_manager.create_table("""
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            text TEXT,
            title TEXT,
            summary TEXT
        )
    """)

    db_manager.create_table("""
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            claim_id TEXT,
            title TEXT,
            url TEXT,
            site TEXT,
            body TEXT,
            topic TEXT,
            entities TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )
    """)

    db_manager.create_table("""
        CREATE TABLE IF NOT EXISTS answers (
            id TEXT PRIMARY KEY,
            claim_id TEXT,
            answer TEXT,
            graphs_folder TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )
    """)

    db_manager.create_table("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            claim_id TEXT,
            predicted_label TEXT,   -- What the AI said (Supported/Refuted/Not Enough Information or error message)
            ground_truth TEXT,      -- The actual truth from the dataset for future implementation

            system_type TEXT,       -- Track the model (FoxAI-GraphRAG, LLM-Only, BM25-RAG, Dense-RAG)
            dataset_setting TEXT,   -- Track the setting (FEVER-Controlled, FEVER-OpenWeb, None for when not using a dataset)
            
            -- Latency Metrics (Seconds)
            latency_preprocessor REAL,
            latency_retrieval REAL,
            latency_generation REAL,
            
            -- Token Metrics
            tokens_preprocessor INTEGER,
            tokens_retrieval INTEGER,
            tokens_generation INTEGER,
            
            -- LLM Call Metrics
            calls_preprocessor INTEGER,
            calls_retrieval INTEGER,
            calls_generation INTEGER,
            
            evidence_log_path TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )
    """)

    return db_manager


if __name__ == "__main__":
    initialize_database_schema()
    logger.info("Database tables verified.")
