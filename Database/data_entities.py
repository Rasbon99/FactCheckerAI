import uuid

from Database.sqldb import Database

from log import Logger

class Claim:
    def __init__(self, text, title, summary, claim_id=None, db=None):
        """
        Initializes a Claim object with text and optionally a provided claim ID. 
        It will also save the claim to the database.

        Args:
            text (str): The text of the claim.
            claim_id (str, optional): The ID of the claim. If not provided, a new UUID is generated.
            db (Database, optional): The database object to use. Defaults to a new Database instance.
        
        Raises:
            Exception: If there is an error during claim creation or database operation.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.db = db if db else Database()  
        self.id = claim_id if claim_id else str(uuid.uuid4())  
        self.text = text
        self.title = title[2:]
        self.summary = summary
        self.logger.info("Creating claim with ID: %s", self.id)
        self.save_to_db()

    def save_to_db(self):
        """
        Saves the claim to the database, creating the table if it doesn't exist.
        After saving, it exports the claims data to a CSV file.

        Raises:
            Exception: If there is an error while saving the claim to the database.
        """
        self.logger.info("Saving claim to the database.")
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT,
                title TEXT,
                summary TEXT
            )
        """
        self.db.create_table(create_table_sql)
        self.db.execute_query("INSERT INTO claims (id, text, title, summary) VALUES (?, ?, ?, ?)", 
                              (self.id, self.text, self.title, self.summary))
        self.logger.info("Claim with ID %s saved to the database.", self.id)
    
    def get_dict_sources(self):
        """
        Retrieves all sources associated with the claim from the database.

        Returns:
            list: A list of dictionaries, each representing a source associated with this claim.
        
        Raises:
            Exception: If there is an error during fetching sources from the database.
        """
        self.logger.info("Fetching sources for claim ID %s.", self.id)
        rows = self.db.fetch_all("SELECT * FROM sources WHERE claim_id = ?", (self.id,))
        sources = [
            {
                "source_id": row['id'],
                "claim_id": row['claim_id'],
                "title": row['title'],
                "url": row['url'],
                "site": row['site'],
                "body": row['body'],
                "topic": row['topic'],
                "entities": row['entities']
                
            }
            for row in rows
        ]
        self.logger.info("Found %d sources for claim ID %s.", len(sources), self.id)
        return sources
    
    def add_sources(self, sources_data):
        """
        Adds multiple sources associated with the claim to the database and exports the data to CSV.

        Args:
            sources_data (list of dict): A list of dictionaries containing source data to insert.
        
        Raises:
            Exception: If there is an error while inserting sources into the database.
        """
        self.logger.info("Adding sources for claim ID %s.", self.id)
        create_table_sql = """
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
        self.db.create_table(create_table_sql)

        # Insert each source into the database
        for data in sources_data:
            self.db.execute_query("""
                INSERT INTO sources (id, claim_id, title, url, site, body, topic, entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), self.id, data['title'], data['url'], data['site'],
                  data['body'], data['topic'], str(data['entities'])))