import json
import os
import dotenv
from groq import Groq
from ollama import Client
from collections import defaultdict

from Preprocessor.ollama_client import OllamaClient

from log import Logger

class NER:
    def __init__(self, env_file="key.env"):
        """
        Initializes the NER class with a specific model and configures the Groq API client.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.client = Groq()
        
        self.ollama_client = OllamaClient()
        if not self.ollama_client.is_running():
            self.ollama_client.start_server()
        
        self.ollama = Client()
        self.model_ollama = os.getenv("MODEL_LLM_NER")

    def extract_entities_and_topic(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Extracts entities and the main topic from the given text using the Groq API.
        """
        self.logger.info("Starting entity and topic extraction process.")
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """you are an NER model that extracts entities and the topic from a text.\n 
                    The output must be strictly formatted as: {\"topic\": \"Technology\", \"entities\": [\"Elon Musk\", \"SpaceX\", \"Tesla\", \"Paris\"]}"""},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop
            )
            
            self.logger.info("Groq API call successful.")
            result = response.choices[0].message.content.strip()
            self.logger.debug("Raw API response: %s", result)

            return json.loads(result)  # Parsing JSON direttamente
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error("Error extracting topic and entities: %s", e)
            return None

    def find_similar_entities_globally(self, entities):
        """
        Finds unified versions of entities by analyzing them in context using Ollama LLM.
        This version passes all entities together to allow the LLM to have a global understanding.
        """
        self.logger.debug(f"Finding similar entities globally for: {entities}")

        try:
            # Combine all entities into a single prompt for Ollama to provide context for each
            input_entities = ", ".join(entities)
            unified_entities = self.ollama.chat(
                model=self.model_ollama,
                messages=[
                    {
                        "role": "system",
                        "content": f"Provide the unified or normalized version for each of these entities: {input_entities}. "
                                "Please return the unified version for each entity in the same order, separated by commas."
                    }
                ],
                options={"temperature": 0.0}
            ).get("message", {}).get("content", "").strip()

            # Split the returned unified entities
            unified_entities_list = unified_entities.split(",")
            unified_entities_list = [ue.strip() for ue in unified_entities_list]

            # Map original entities to their unified versions
            if len(unified_entities_list) != len(entities):
                raise ValueError("The number of unified entities does not match the number of input entities.")

            unified_mapping = {entities[i]: unified_entities_list[i] for i in range(len(entities))}

            # Group entities by their unified version
            entity_groups = defaultdict(list)
            for entity, unified in unified_mapping.items():
                entity_groups[unified].append(entity)

            self.logger.debug(f"Grouped entities globally: {dict(entity_groups)}")

            return entity_groups

        except Exception as e:
            self.logger.error(f"Error in global entity similarity analysis: {e}")
            # Fallback: return each entity as its own group
            return {entity: [entity] for entity in entities}

    def merge_entities(self, sources):
        """
        Unifies similar entities across sources by replacing them with a unified version.
        """
        self.logger.info("Starting to merge entities from sources.")

        # Extract all unique entities from sources
        raw_entities = list(set(entity for source in sources for entity in source.get("entities", [])))
        self.logger.debug(f"Filtered unique raw entities: {raw_entities}")

        # Group similar entities globally using Ollama
        entity_groups = self.find_similar_entities_globally(raw_entities)

        # Create a mapping of original entities to their unified versions
        unified_mapping = {}
        for unified, originals in entity_groups.items():
            for original in originals:
                unified_mapping[original] = unified
        self.logger.info(f"Unified mapping of entities: {unified_mapping}")

        # Update sources with unified entities
        for source in sources:
            updated_entities = []
            for entity in source.get("entities", []):
                if entity in unified_mapping:
                    updated_entities.append(unified_mapping[entity])
                    self.logger.debug(f"Replaced '{entity}' with '{unified_mapping[entity]}' in source.")
                else:
                    updated_entities.append(entity)
            source["entities"] = list(set(updated_entities))  # Remove duplicates

        self.logger.info("Entities merged and sources updated successfully.")
        return sources