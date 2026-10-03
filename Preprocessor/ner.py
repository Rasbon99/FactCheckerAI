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
        dotenv.load_dotenv(env_file, override=False)
        self.model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.client = Groq()

    def extract_entities_and_topic(
        self, text, max_tokens=1024, temperature=0.0, stop=None
    ):
        """
        Extracts entities and the main topic from the given text using the Groq API.

        Args:
            text (str): The text from which entities and the topic will be extracted.
            max_tokens (int, optional): The maximum number of tokens for the response. Default is 1024.
            temperature (float, optional): Controls randomness in the model output. Default is 0.0.
            stop (list, optional): A list of stop sequences for the model to terminate at. Default is None.

        Returns:
            tuple: (dict containing topic/entities, tokens_used)
        """
        self.logger.info("Starting entity and topic extraction process.")
        tokens = 0

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an NER model that extracts entities and the topic from a text.
                        The output MUST be a valid JSON object strictly formatted as:
                        {"topic": "Technology", "entities": ["Elon Musk", "SpaceX", "Tesla", "Paris"]}""",
                    },
                    {"role": "user", "content": text},
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
                response_format={"type": "json_object"},
            )

            self.logger.info("Groq API call successful.")
            tokens = response.usage.total_tokens if response.usage else 0

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("API response content is None")

            result = content.strip()
            self.logger.debug("Raw API response: %s", result)

            return json.loads(result), tokens

        except Exception as e:
            self.logger.exception("Error extracting topic and entities: %s", e)
            raise

    def find_similar_entities_globally(
        self, entities, max_tokens=1024, temperature=0.0, stop=None
    ):
        """
        Finds unified versions of entities by analyzing them in context using GroqCloud LLM.

        Args:
            entities (list): A list of entities to find unified versions for.

        Returns:
            tuple: (dict mapping unified names to originals, tokens_used)
        """
        self.logger.debug("Finding similar entities globally...")
        tokens = 0

        try:
            system_prompt = f"""Please normalize or unify the following list of entities:
            {json.dumps(entities)}

            You must output a single JSON object where the keys are the EXACT original entity names,
            and the values are the single unified version of that entity.
            If an entity has multiple valid representations or acronyms, select the most common form.
            If an entity is already unified, map it to itself.
            Do not include any extra information. Output ONLY the JSON object.

            Example output format:
            {{"U.S.A.": "United States", "USA": "United States", "Apple Inc": "Apple", "Elon Musk": "Elon Musk"}}"""

            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
                response_format={"type": "json_object"},
            )

            tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content
            self.logger.debug(f"Response content: {response_content}")

            if not response_content:
                raise RuntimeError("API response content is None")

            unified_mapping = json.loads(response_content)

            entity_groups = defaultdict(list)
            for original_entity, unified_entity in unified_mapping.items():
                # Failsafe in case the LLM dropped an entity
                if original_entity in entities:
                    entity_groups[unified_entity].append(original_entity)

            # Catch any entities the LLM might have accidentally left out of the JSON
            for original_entity in entities:
                if original_entity not in unified_mapping:
                    entity_groups[original_entity].append(original_entity)

            self.logger.debug(f"Grouped entities globally: {dict(entity_groups)}")
            return dict(entity_groups), tokens

        except Exception as e:
            self.logger.exception(f"Error in global entity similarity analysis: {e}")
            raise

    def merge_entities(self, sources):
        """
        Unifies similar entities across sources by replacing them with a unified version.

        Args:
            sources (list): A list of source dictionaries containing entities to be merged.

        Returns:
            tuple: (list of sources with unified entities, tokens_used)
        """
        self.logger.info("Starting to merge entities from sources.")

        raw_entities = list(
            set(entity for source in sources for entity in source.get("entities", []))
        )
        self.logger.debug(f"Filtered unique raw entities: {raw_entities}")

        entity_groups, tokens = self.find_similar_entities_globally(raw_entities)

        unified_mapping = {}
        for unified, originals in entity_groups.items():
            for original in originals:
                unified_mapping[original] = unified
        self.logger.info(f"Unified mapping of entities: {unified_mapping}")

        for source in sources:
            updated_entities = []
            for entity in source.get("entities", []):
                if entity in unified_mapping and entity != unified_mapping[entity]:
                    updated_entities.append(unified_mapping[entity])
                    self.logger.debug(
                        f"Replaced '{entity}' with '{unified_mapping[entity]}' in source."
                    )
                else:
                    updated_entities.append(entity)
            source["entities"] = list(set(updated_entities))

        self.logger.info("Entities merged and sources updated successfully.")
        return sources, tokens
