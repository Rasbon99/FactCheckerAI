import uuid
import os
import json
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

        # Use the provided claim_id, or generate a new one if None is passed
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
        self.db.execute_query(
            "INSERT INTO claims (id, text, title, summary) VALUES (?, ?, ?, ?)",
            (self.id, self.text, self.title, self.summary),
        )
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
                "source_id": row["id"],
                "claim_id": row["claim_id"],
                "title": row["title"],
                "url": row["url"],
                "site": row["site"],
                "body": row["body"],
                "topic": row["topic"],
                "entities": row["entities"],
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

        # Insert each source into the database
        for data in sources_data:
            self.db.execute_query(
                """
                INSERT INTO sources (id, claim_id, title, url, site, body, topic, entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(uuid.uuid4()),
                    self.id,
                    data["title"],
                    data["url"],
                    data["site"],
                    data["body"],
                    data["topic"],
                    str(data["entities"]),
                ),
            )
        self.logger.info(
            "Added %d sources for claim ID %s.", len(sources_data), self.id
        )

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


class Answer:
    def __init__(self, claim_id, answer, graphs_folder, answer_id=None, db=None):
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
        self.graphs_folder = graphs_folder
        self.save_to_db()

    def save_to_db(self):
        """
        Saves the answer to the database, creating the table if it doesn't exist.
        After saving, it exports the answers data to a CSV file.

        Raises:
            Exception: If there is an error while saving the answer to the database.
        """
        self.db.execute_query(
            "INSERT INTO answers (id, claim_id, answer, graphs_folder) VALUES (?,?,?,?)",
            (self.id, self.claim_id, self.answer, self.graphs_folder),
        )


class Experiment:

    def __init__(
        self,
        claim_id,
        predicted_label,
        ground_truth,
        latencies,
        tokens,
        calls,
        evidence_data,
        system_type="FoxAI-GraphRAG",
        dataset_setting="User-Query",
        experiment_id=None,
        db=None,
    ):
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.db = db if db else Database()
        self.id = experiment_id if experiment_id else str(uuid.uuid4())
        self.claim_id = claim_id
        self.predicted_label = predicted_label
        self.ground_truth = ground_truth

        self.system_type = system_type
        self.dataset_setting = dataset_setting

        # Store the metric dictionaries
        self.latencies = latencies
        self.tokens = tokens
        self.calls = calls

        # Save evidence and get the file path
        self.evidence_log_path = self.save_evidence(evidence_data)

        # Save metrics to SQLite
        self.save_to_db()

    def save_evidence(self, evidence_data):
        self.logger.info("Saving raw evidence to JSON for experiment ID %s.", self.id)

        # Pull the path directly from the environment variable
        log_dir = os.environ.get("EXPERIMENTS_PATH", "data/experiments")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            self.logger.info("Created experiments directory: %s", log_dir)

        file_path = os.path.join(log_dir, f"{self.id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, indent=4)

        return file_path

    def save_to_db(self):
        self.db.execute_query(
            """INSERT INTO experiments 
               (id, claim_id, predicted_label, ground_truth, system_type, dataset_setting, 
                latency_preprocessor, latency_retrieval, latency_graph_rag, 
                tokens_preprocessor, tokens_retrieval, tokens_graph_rag,
                calls_preprocessor, calls_retrieval, calls_graph_rag,
                evidence_log_path) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.id,
                self.claim_id,
                self.predicted_label,
                self.ground_truth,
                self.system_type,
                self.dataset_setting,
                # Latencies
                self.latencies.get("preprocessor", 0.0),
                self.latencies.get("retrieval", 0.0),
                self.latencies.get("graph_rag", 0.0),
                # Tokens
                self.tokens.get("preprocessor", 0),
                self.tokens.get("retrieval", 0),
                self.tokens.get("graph_rag", 0),
                # Calls
                self.calls.get("preprocessor", 0),
                self.calls.get("retrieval", 0),
                self.calls.get("graph_rag", 0),
                self.evidence_log_path,
            ),
        )
        self.logger.info("Experiment ID %s saved.", self.id)
