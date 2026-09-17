import dotenv

from Preprocessor.ner import NER
from Preprocessor.summarizer import Summarizer

from log import Logger


class Preprocessing_Pipeline:
    def __init__(self, env_file="key.env", config=None):
        """
        Initializes the preprocessing pipeline, setting up the necessary components like NER, Summarizer, and configuration.

        Args:
            env_file (str, optional): The environment file containing API keys. Default is "key.env".
            config (dict, optional): Configuration options for summarization and NER.
                                      Default is {"summarize": True, "NER": True}.
        """
        dotenv.load_dotenv(env_file, override=False)

        self.logger = Logger(self.__class__.__name__).get_logger()
        self.ner = NER()
        self.summarizer = Summarizer()

        self.config = {"summarize": True, "NER": True}
        if config:
            self.config.update(config)

    def run_claim_pipe(self, claim, max_lenght=150):
        """
        Processes a claim by generating a search title and a summary.

        Args:
            claim (str): The claim text to preprocess.
            max_lenght (int, optional): The maximum length for the summary. Default is 150.

        Returns:
            tuple: ((claim_title, claim_summary), token_data)
        """
        self.logger.info("Starting claim preprocessing...")
        token_data = {"total": 0, "calls": 0}

        if self.config.get("summarize", True):
            claim_title, title_tokens = self.summarizer.claim_title_summarize(
                claim, max_lenght
            )
            token_data["calls"] += 1
            token_data["total"] += title_tokens

            claim_summary, summary_tokens = self.summarizer.generate_summary(
                claim, max_lenght
            )
            token_data["calls"] += 1
            token_data["total"] += summary_tokens

            self.logger.info("Claim preprocessing completed.")
            return (claim_title, claim_summary), token_data

        self.logger.info("Claim preprocessing completed.")
        return (claim, claim), token_data

    def run_sources_pipe(self, sources, max_lenght=1024):
        """
        Processes a list of sources by summarizing each source and extracting entities/topics.

        Args:
            sources (list): A list of source objects (e.g., text articles) to preprocess.
            max_lenght (int, optional): The maximum token length for the summary. Default is 1024.

        Returns:
            tuple: (list of preprocessed sources, token_data)
        """
        self.logger.info("Starting sources preprocessing...")
        token_data = {"total": 0, "calls": 0}

        if self.config.get("summarize", True):
            new_bodies, sum_tokens = self.summarizer.summarize_texts(
                [d["body"] for d in sources], max_lenght
            )

            token_data["total"] += sum_tokens
            token_data["calls"] += len(sources)

            for d, new_body in zip(sources, new_bodies):
                if new_body:
                    d["body"] = new_body

        if self.config.get("NER", True):
            for source in sources:
                topic_and_entities, tokens = self.ner.extract_entities_and_topic(
                    source["body"]
                )
                token_data["calls"] += 1
                token_data["total"] += tokens

                if topic_and_entities is None:
                    topic_and_entities = {"topic": None, "entities": []}

                source["topic"] = topic_and_entities["topic"]
                source["entities"] = topic_and_entities["entities"]

            sources, merge_tokens = self.ner.merge_entities(sources)
            token_data["calls"] += 1
            token_data["total"] += merge_tokens

        self.logger.info("Sources preprocessing completed.")
        return sources, token_data
