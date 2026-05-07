import sqlite3
import os
import dotenv
import time
from log import Logger

dotenv.load_dotenv("key.env", override=True)
WIKI_DB_PATH = os.getenv("FEVER_WIKIPEDIA_DB_PATH", "Datasets/fever_wiki.db")
logger = Logger(__name__).get_logger()


def build_fts_index():
    logger.info(f"Connecting to Wikipedia Database at: {WIKI_DB_PATH}")

    if not os.path.exists(WIKI_DB_PATH):
        logger.error("Database not found!")
        return

    conn = sqlite3.connect(WIKI_DB_PATH)
    cursor = conn.cursor()

    logger.info("Checking if FTS5 Virtual Table already exists...")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='wiki_fts'"
    )
    if cursor.fetchone():
        logger.info(
            "✅ The 'wiki_fts' index already exists! You are ready to run the baseline."
        )
        conn.close()
        return

    logger.info("Building the BM25 (FTS5) Search Index...")
    logger.info(
        "⏳ This involves mathematically indexing 5GB of text. It may take 5-10 minutes."
    )
    logger.info("Make sure your Mac is plugged into power!")

    start_time = time.time()

    try:
        # 1. Create the high-performance Virtual Table
        logger.info("1/3 Creating virtual table...")
        cursor.execute("""
            CREATE VIRTUAL TABLE wiki_fts USING fts5(
                page_id, 
                lines, 
                tokenize = 'porter' 
            )
        """)

        # 2. Populate it with the data from your main table
        logger.info("2/3 Copying and indexing 5.4 million articles (Please wait...)")
        cursor.execute(
            "INSERT INTO wiki_fts (page_id, lines) SELECT page_id, lines FROM wiki_articles"
        )

        # 3. Commit the massive transaction
        logger.info("3/3 Saving the index to disk...")
        conn.commit()

        elapsed = (time.time() - start_time) / 60
        logger.info(f"SUCCESS! BM25 Index built in {elapsed:.2f} minutes.")

    except Exception as e:
        logger.error(f"Error building index: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    build_fts_index()
