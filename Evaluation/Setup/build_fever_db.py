import os
import sqlite3
import json
import glob
import time
import dotenv
from log import Logger

dotenv.load_dotenv("key.env", override=True)

WIKIPEDIA_FOLDER = os.getenv("FEVER_WIKIPEDIA_PAGES_PATH", "Datasets/wiki-pages")
DB_PATH = os.getenv("FEVER_WIKIPEDIA_DB_PATH", "Datasets/fever_wiki.db")
logger = Logger(__name__).get_logger()


def build_wikipedia_database():
    logger.info(
        f"Starting Phase 1: Building Wikipedia Database from {WIKIPEDIA_FOLDER}..."
    )

    # 1. Create a dedicated SQLite database just for the Wikipedia dump
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create a table with 'page_id' as the Primary Key for lightning-fast lookups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wiki_articles (
            page_id TEXT PRIMARY KEY,
            lines TEXT
        )
    """)

    # 2. Find all the .jsonl files in the folder
    jsonl_files = glob.glob(os.path.join(WIKIPEDIA_FOLDER, "*.jsonl"))
    if not jsonl_files:
        logger.error(f"Could not find any .jsonl files in {WIKIPEDIA_FOLDER}")
        return

    logger.info(f"Found {len(jsonl_files)} JSONL files to process.")

    total_articles = 0
    start_time = time.time()

    # 3. Read each file and insert the data in bulk (much faster!)
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}...")

        articles_to_insert = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                page_id = data.get("id", "")
                lines_text = data.get("lines", "")

                # Only save articles that actually have text
                if page_id and lines_text:
                    articles_to_insert.append((page_id, lines_text))

        # Bulk insert into SQLite (IGNORE if the page already exists somehow)
        cursor.executemany(
            """
            INSERT OR IGNORE INTO wiki_articles (page_id, lines)
            VALUES (?, ?)
        """,
            articles_to_insert,
        )

        conn.commit()
        total_articles += len(articles_to_insert)

    conn.close()

    end_time = time.time()
    logger.info("=" * 50)
    logger.info("DATABASE BUILD COMPLETE!")
    logger.info(f"Total Wikipedia Pages Indexed: {total_articles:,}")
    logger.info(f"Time Taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Database saved to: {DB_PATH}")
    logger.info("=" * 50)


if __name__ == "__main__":
    build_wikipedia_database()
