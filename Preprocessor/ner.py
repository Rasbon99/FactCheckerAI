import json
import os
import dotenv
from groq import Groq
from ollama import Client
from log import Logger
from collections import defaultdict
from Preprocessor.ollama_client import OllamaClient

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
        self.model_ollama = "MODEL_LLM_NER"

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

    def normalize_entity(self, entity):
        """
        Optimized method to normalize a single entity name using Ollama.
        """
        self.logger.debug(f"Normalizing entity: {entity}")
        try:
            response = self.ollama.chat(
                model=self.model_ollama,
                messages=[{"role": "system", "content": f"Normalize this entity: '{entity}'. Return only strictly the entity without comments"}],
                options={"temperature":0.5}
            )
            normalized = response.get('message', {}).get('content', '').strip()
            if not normalized:
                raise ValueError("Empty response from Ollama")
            self.logger.debug(f"Entity '{entity}' normalized to '{normalized}'")
            return normalized
        except Exception as e:
            self.logger.error(f"Error normalizing entity '{entity}': {e}")
            return entity  # Return original entity on failure

    def merge_entities(self, sources):
        """
        Unifies similar entities using Ollama and updates sources by replacing similar entities with a unified version.
        """
        self.logger.info("Starting merging entities from sources.")
        
        # Extract all entities from sources
        raw_entities = [entity for source in sources for entity in source.get("entities", [])]
        self.logger.debug(f"Extracted raw entities: {raw_entities}")
        
        # Group similar entities using Ollama
        entity_groups = defaultdict(list)
        for entity in raw_entities:
            normalized_entity = self.normalize_entity(entity)
            entity_groups[normalized_entity.lower()].append(entity)
        self.logger.debug(f"Grouped entities: {dict(entity_groups)}")
        
        # Keep unified entity names from Ollama
        unified_entities = {key: self.normalize_entity(key) for key in entity_groups if len(entity_groups[key]) > 1}
        self.logger.info(f"Unified entities mapping: {unified_entities}")
        
        # Update sources with unified entities
        for source in sources:
            updated_entities = []
            for entity in source.get("entities", []):
                lower_entity = entity.lower()
                if lower_entity in unified_entities:
                    updated_entities.append(unified_entities[lower_entity])
                    self.logger.debug(f"Replaced '{entity}' with '{unified_entities[lower_entity]}' in source.")
                else:
                    updated_entities.append(entity)
            source["entities"] = list(set(updated_entities))  # Ensure unique entities per source
        
        self.logger.info("Entities merged and sources updated successfully.")
        return sources
