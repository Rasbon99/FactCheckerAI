import csv
import dotenv
import os

dotenv.load_dotenv("key.env", override=True)

class CSVExporter:
    @staticmethod
    def export_claims(db=os.getenv("SQLDB_PATH"), filename=os.getenv("CLAIM_CSV_PATH")):
        claims = db.fetch_all("SELECT * FROM claims")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'text'])
            for claim in claims:
                writer.writerow([claim['id'], claim['text']])

    @staticmethod
    def export_sources(db=os.getenv("SQLDB_PATH"), filename=os.getenv("SOURCES_CSV_PATH")):
        sources = db.fetch_all("SELECT * FROM sources")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'claim_id', 'title', 'url', 'site', 'body', 'entities', 'topic'])
            for source in sources:
                writer.writerow([source['id'], source['claim_id'], source['title'], source['url'],
                                 source['site'], source['body'], source['entities'], source['topic']])
