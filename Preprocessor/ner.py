import json
import os
import dotenv
from groq import Groq
from collections import defaultdict

from log import Logger

class NER:
    def __init__(self, env_file="key.env"):
        """
        Initializes the NER class with a specific model and configures the Groq API client.

        Args:
            env_file (str, optional): The path to the environment file containing API keys. Default is "key.env".
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.client = Groq()

    def extract_entities_and_topic(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Extracts entities and the main topic from the given text using the Groq API.

        Args:
            text (str): The text from which entities and the topic will be extracted.
            max_tokens (int, optional): The maximum number of tokens for the response. Default is 1024.
            temperature (float, optional): Controls randomness in the model output. Default is 0.5.
            stop (list, optional): A list of stop sequences for the model to terminate at. Default is None.

        Returns:
            dict: A dictionary containing the topic and a list of entities extracted from the text.
        
        Raises:
            json.JSONDecodeError: If the API response cannot be parsed as JSON.
            Exception: If an error occurs during the API call or entity extraction.
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

            return json.loads(result)  # Parsing JSON directly
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error("Error extracting topic and entities: %s", e)
            return None

    def find_similar_entities_globally(self, entities, max_tokens=1024, temperature=0.3, stop=None):
        """
        Finds unified versions of entities by analyzing them in context using GroqCloud LLM.

        Args:
            entities (list): A list of entities to find unified versions for.

        Returns:
            dict: A dictionary mapping unified entity names to their original variants.
        
        Raises:
            Exception: If there is an error during entity normalization.
        """
        self.logger.debug(f"Finding similar entities globally...")

        try:
            input_entities = ", ".join(entities)
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""Please normalize or unify the following entities: {input_entities}. For each entity, provide a single unified version. 
                                    If an entity has multiple valid representations, variations, synonyms or acronyms, choose the most common or widely recognized one. 
                                    Please return the unified version for each entity in the same order, separated by commas. 
                                    If any of the entities are already unified or do not require normalization, simply return them as they are.
                                    Do not include any additional information, note or context in the response.
                                    Example: input: ['United States', 'USA', 'US', 'U.S.'] output: ['United States', 'United States', 'United States', 'United States']"""
                    }
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop
            )
            
            # Extraction from response
            response_content = response.choices[0].message.content
            self.logger.debug(f"Response content: {response_content}")
            
            # Handling the response carefully
            unified_entities_list = [ue.strip() for ue in response_content.split(',')]

            # Ensure the number of unified entities matches the input entities count
            if len(unified_entities_list) != len(entities):
                raise ValueError("The number of unified entities does not match the number of input entities.")

            # Map original entities to their unified versions
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

        Args:
            sources (list): A list of source dictionaries containing entities to be merged.

        Returns:
            list: A list of sources with unified entities.
        
        Raises:
            Exception: If there is an error during the merging process.
        """
        self.logger.info("Starting to merge entities from sources.")

        raw_entities = list(set(entity for source in sources for entity in source.get("entities", [])))
        self.logger.debug(f"Filtered unique raw entities: {raw_entities}")

        entity_groups = self.find_similar_entities_globally(raw_entities)

        unified_mapping = {}
        for unified, originals in entity_groups.items():
            for original in originals:
                unified_mapping[original] = unified
        self.logger.info(f"Unified mapping of entities: {unified_mapping}")

        # Merge the entities across the sources
        for source in sources:
            updated_entities = []
            for entity in source.get("entities", []):
                if entity in unified_mapping and entity != unified_mapping[entity]:
                    updated_entities.append(unified_mapping[entity])
                    self.logger.debug(f"Replaced '{entity}' with '{unified_mapping[entity]}' in source.")
                else:
                    updated_entities.append(entity)
            source["entities"] = list(set(updated_entities))  # Remove duplicates

        self.logger.info("Entities merged and sources updated successfully.")
        return sources