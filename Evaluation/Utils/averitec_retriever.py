import os
import json


class AVeriTeCKnowledgeRetriever:
    def __init__(self, knowledge_store_dir=None):
        """
        Initializes the retriever. If no directory is provided, it attempts to
        load it from the environment variables.
        """
        if knowledge_store_dir is None:
            self.knowledge_store_dir = os.getenv(
                "AVERITEC_KNOWLEDGE_STORE_PATH", "Datasets/AVeriTeC/dev_knowledge_store"
            )
        else:
            self.knowledge_store_dir = knowledge_store_dir

    def get_evidence_for_claim(self, claim_id):
        """
        Instantly retrieves the scraped web pages for a specific AVeriTeC claim.
        Returns a list of all sentences scraped for that claim.
        """
        # In AVeriTeC, the file name is literally just [claim_id].json
        filepath = os.path.join(self.knowledge_store_dir, f"{claim_id}.json")

        all_sentences = []

        # If the file doesn't exist, it means the researchers found 0 Google results for this claim
        if not os.path.exists(filepath):
            return all_sentences

        # The file has a .json extension but is actually formatted as JSON Lines (.jsonl)
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # Grab the array of scraped sentences
                    sentences = data.get("url2text", [])
                    all_sentences.extend(sentences)
                except json.JSONDecodeError:
                    # Safely ignore any weird invisible characters that break the parser
                    continue

        return all_sentences
