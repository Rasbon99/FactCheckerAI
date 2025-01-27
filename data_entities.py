from sqldb import Database
from csv_exporter import CSVExporter
import uuid

class Claim:
    def __init__(self, text, claim_id=None, db=None):
        self.db = db if db else Database()  
        self.id = claim_id if claim_id else str(uuid.uuid4())  
        self.text = text
        self.save_to_db()

    def save_to_db(self):
        # Crea la tabella se non esiste
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT
            )
        """
        self.db.create_table(create_table_sql)
        # Inserisci il claim nel database
        self.db.execute_query("INSERT INTO claims (id, text) VALUES (?, ?)", (self.id, self.text))

        # Esporta i claim in CSV dopo l'inserimento
        CSVExporter.export_claims(self.db)

    def get_sources(self):
        rows = self.db.fetch_all("SELECT * FROM sources WHERE claim_id = ?", (self.id,))
        return [Source(
            claim_id=row['claim_id'],
            title=row['title'],
            url=row['url'],
            site=row['site'],
            body=row['body'],
            entities=row['entities'],
            topic=row['topic'],
            source_id=row['id']
        ) for row in rows]

    def add_sources(self, sources_data):
        # Crea la tabella delle fonti se non esiste
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                claim_id TEXT,
                title TEXT,
                url TEXT,
                site TEXT,
                body TEXT,
                entities TEXT,
                topic TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims(id)
            )
        """
        self.db.create_table(create_table_sql)

        # Aggiungi ciascuna fonte al database
        for data in sources_data:
            self.db.execute_query("""
                INSERT INTO sources (id, claim_id, title, url, site, body, entities, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), self.id, data['title'], data['url'], data['site'],
                  data['body'], data['entities'], data['topic']))

        # Esporta le fonti in CSV dopo averle aggiunte
        CSVExporter.export_sources(self.db)


class Source:
    def __init__(self, claim_id, title, url, site, body, entities, topic, source_id=None, db=None):
        self.db = db if db else Database()
        self.id = source_id if source_id else str(uuid.uuid4())
        self.claim_id = claim_id
        self.title = title
        self.url = url
        self.site = site
        self.body = body
        self.entities = entities
        self.topic = topic

    @staticmethod
    def load_all(db=None):
        db = db if db else Database()
        rows = db.fetch_all("SELECT * FROM sources")
        return [Source(
            claim_id=row['claim_id'],
            title=row['title'],
            url=row['url'],
            site=row['site'],
            body=row['body'],
            entities=row['entities'],
            topic=row['topic'],
            source_id=row['id'],
            db=db
        ) for row in rows]
