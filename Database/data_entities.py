import uuid

from Database.sqldb import Database

from log import Logger

from PIL import Image
import io

class Claim:
    def __init__(self, text, claim_id=None, db=None):
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
        self.logger.info("Creating claim with ID: %s", self.id)
        self.save_to_db()
        self.db.execute_query("CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, claim_id TEXT, answer TEXT, image BLOB, FOREIGN KEY (claim_id) REFERENCES claims(id))")

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
                text TEXT
            )
        """
        self.db.create_table(create_table_sql)
        self.db.execute_query("INSERT INTO claims (id, text) VALUES (?, ?)", (self.id, self.text))
        self.logger.info("Claim with ID %s saved to the database.", self.id)

    def get_sources(self):
        """
        Retrieves all sources associated with the claim from the database.

        Returns:
            list: A list of Source objects associated with this claim.
        
        Raises:
            Exception: If there is an error during fetching sources from the database.
        """
        self.logger.info("Fetching sources for claim ID %s.", self.id)
        rows = self.db.fetch_all("SELECT * FROM sources WHERE claim_id = ?", (self.id,))
        sources = [Source(
            claim_id=row['claim_id'],
            title=row['title'],
            url=row['url'],
            site=row['site'],
            body=row['body'],
            score=row['score'],
            topic=row['topic'],
            entities=row['entities'],
            source_id=row['id']
        ) for row in rows]
        self.logger.info("Found %d sources for claim ID %s.", len(sources), self.id)
        return sources
    
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
                "score": row['score'],
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
                score FLOAT,
                topic TEXT,
                entities TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims(id)
            )
        """
        self.db.create_table(create_table_sql)

        # Insert each source into the database
        for data in sources_data:
            self.db.execute_query("""
                INSERT INTO sources (id, claim_id, title, url, site, body, score, topic, entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), self.id, data['title'], data['url'], data['site'],
                  data['body'], data['score'], data['topic'], str(data['entities'])))
        self.logger.info("Added %d sources for claim ID %s.", len(sources_data), self.id)
    def get_answer(self):
        """
        Retrieves the answer associated with the claim from the database.

        Returns:
            Answer: The answer object associated with this claim.
        
        Raises:
            Exception: If there is an error during fetching the answer from the database.
        """
        self.logger.info("Fetching answer for claim ID %s.", self.id)
        row = self.db.fetch_one("SELECT * FROM answers WHERE claim_id = ?", (self.id,))
        if row:
            self.logger.info("Found answer for claim ID %s.", self.id)
            return row['answer'], Image.open(io.BytesIO(row['image']))
        self.logger.info("No answer found for claim ID %s.", self.id)
        return None
    
    def clear_database(self):
        """
        Clears all data associated with the claim from the database.

        Raises:
            Exception: If there is an error during clearing the claim data from the database.
        """
        self.logger.info("Clearing claim data for claim ID %s.", self.id)
        self.db.execute_query("DELETE FROM claims WHERE id = ?", (self.id,))
        self.db.execute_query("DELETE FROM sources WHERE claim_id = ?", (self.id,))
        self.db.execute_query("DELETE FROM answers WHERE claim_id = ?", (self.id,))
        self.logger.info("Claim data for claim ID %s cleared.", self.id)
    
    def has_answer(self):
        """
        Checks if the claim has an answer.

        Returns:
            bool: True if the claim has an answer, False otherwise.
        """
        self.logger.info("Checking if claim ID %s has an answer.", self.id)
        row = self.db.fetch_one("SELECT * FROM answers WHERE claim_id = ?", (self.id,))
        return row is not None

class Source:
    def __init__(self, claim_id, title, url, site, body, score, topic, entities, source_id=None, db=None):
        """
        Initializes a Source object with details about a specific source.

        Args:
            claim_id (str): The ID of the associated claim.
            title (str): The title of the source.
            url (str): The URL of the source.
            site (str): The site of the source.
            body (str): The body/content of the source.
            entities (str): The entities mentioned in the source.
            topic (str): The topic of the source.
            source_id (str, optional): The ID of the source. If not provided, a new UUID is generated.
            db (Database, optional): The database object to use. Defaults to a new Database instance.
        
        Raises:
            Exception: If there is an error during source object initialization.
        """
        self.db = db if db else Database()
        self.id = source_id if source_id else str(uuid.uuid4())
        self.claim_id = claim_id
        self.title = title
        self.url = url
        self.site = site
        self.body = body
        self.score = score
        self.topic = topic
        self.entities = entities

    @staticmethod
    def load_all(db=None):
        """
        Loads all source records from the database.

        Args:
            db (Database, optional): The database object to use. Defaults to a new Database instance.

        Returns:
            list: A list of Source objects loaded from the database.
        
        Raises:
            Exception: If there is an error while fetching all sources from the database.
        """
        db = db if db else Database()
        rows = db.fetch_all("SELECT * FROM sources")
        return [Source(
            claim_id=row['claim_id'],
            title=row['title'],
            url=row['url'],
            site=row['site'],
            body=row['body'],
            score=row['score'],
            topic=row['topic'],
            entities=row['entities'],
            source_id=row['id'],
            db=db
        ) for row in rows]
        

class Answer():
    def __init__(self, claim_id, answer, answer_id=None, db=None,image=None):
        """
        Initializes an Answer object with details about an answer to a specific claim.

        Args:
            claim_id (str): The ID of the associated claim.
            answer (str): The answer to the claim.
            answer_id (str, optional): The ID of the answer. If not provided, a new UUID is generated.
            db (Database, optional): The database object to use. Defaults to a new Database instance.
        
        Raises:
            Exception: If there is an error during answer object initialization.
        """
        self.db = db if db else Database()
        self.id = answer_id if answer_id else str(uuid.uuid4())
        self.claim_id = claim_id
        self.answer = answer
        self.image = image
        self.save_to_db()
    def set_answer(self, answer):
        """
        Sets the answer for the claim.

        Args:
            answer (str): The answer to the claim.
        """
        self.answer = answer
    
    def set_image(self, image):
        """
        Sets the image for the answer.

        Args:
            image (Image object from PIL): The image data to set.
        """
        self.image = image

    def save_to_db(self):
        """
        Saves the answer to the database, creating the table if it doesn't exist.
        After saving, it exports the answers data to a CSV file.

        Raises:
            Exception: If there is an error while saving the answer to the database.
        """
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS answers (
                id TEXT PRIMARY KEY,
                claim_id TEXT,
                answer TEXT,
                image BLOB,
                FOREIGN KEY (claim_id) REFERENCES claims(id)
            )
        """
        self.db.create_table(create_table_sql)
        
        if self.image:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
        
        self.db.execute_query("INSERT INTO answers (id, claim_id, answer,image) VALUES (?,?,?,?)",
                              (self.id, self.claim_id, self.answer, img_byte_arr))