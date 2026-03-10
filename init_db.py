from Database.sqldb import Database


def initialize_database_schema(db: Database | None = None):
    """
    Creates all required SQLite tables if they do not already exist.

    Args:
        db (Database, optional): Existing Database instance. If omitted, a new instance is created.

    Returns:
        Database: The Database instance used for initialization.
    """
    db_manager = db if db is not None else Database()

    db_manager.create_table(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            text TEXT,
            title TEXT,
            summary TEXT
        )
    """
    )

    db_manager.create_table(
        """
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
    """
    )

    db_manager.create_table(
        """
        CREATE TABLE IF NOT EXISTS answers (
            id TEXT PRIMARY KEY,
            claim_id TEXT,
            answer TEXT,
            graphs_folder TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )
    """
    )

    db_manager.create_table(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            claim_id TEXT,
            latency_preprocessor REAL,
            latency_retrieval REAL,
            latency_graph_rag REAL,
            total_tokens INTEGER,
            llm_calls INTEGER,
            evidence_log_path TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )
    """
    )

    return db_manager


if __name__ == "__main__":
    initialize_database_schema()
    print("✅ Database tables verified.")
