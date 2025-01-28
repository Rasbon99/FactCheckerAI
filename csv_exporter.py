import csv
import dotenv
import os
from log import Logger

dotenv.load_dotenv("key.env", override=True)

class CSVExporter:
    @staticmethod
    def export_claims(db=os.getenv("SQLDB_PATH"), filename=os.getenv("CLAIM_CSV_PATH")):
        """
        Exports the claims data from the database to a CSV file.

        Args:
            db (Database): The database connection object used to fetch claims data.
            filename (str): The path where the CSV file will be saved. Default is loaded from the environment variable.
        """
        logger = Logger(CSVExporter.__name__).get_logger()
        logger.info("Starting claims export to CSV.")

        claims = db.fetch_all("SELECT * FROM claims")
        logger.info("Fetched %d claims from the database.", len(claims))

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'text'])
            for claim in claims:
                writer.writerow([claim['id'], claim['text']])

        logger.info("Claims export to %s completed successfully.", filename)

    @staticmethod
    def export_sources(db=os.getenv("SQLDB_PATH"), filename=os.getenv("SOURCES_CSV_PATH")):
        """
        Exports the sources data from the database to a CSV file.

        Args:
            db (Database): The database connection object used to fetch source data.
            filename (str): The path where the CSV file will be saved. Default is loaded from the environment variable.
        """
        logger = Logger(CSVExporter.__name__).get_logger()
        logger.info("Starting sources export to CSV.")

        sources = db.fetch_all("SELECT * FROM sources")
        logger.info("Fetched %d sources from the database.", len(sources))

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'claim_id', 'title', 'url', 'site', 'body', 'entities', 'topic'])
            for source in sources:
                writer.writerow([source['id'], source['claim_id'], source['title'], source['url'],
                                 source['site'], source['body'], source['entities'], source['topic']])

        logger.info("Sources export to %s completed successfully.", filename)
