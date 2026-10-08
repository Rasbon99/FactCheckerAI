import json
import os
import dotenv
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from llamacpp_client import ChatLlamaCppServer

from log import Logger


class NER:
    def __init__(self, env_file="key.env"):
        """
        Initializes the NER class with a specific model and configures the llama.cpp client.

        Args:
            env_file (str, optional): The path to the environment file containing configuration. Default is "key.env".
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=False)
        self.model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")

    def extract_entities_and_topic(
        self, text, max_tokens=1024, temperature=0.0, stop=None
    ):
        """
        Extracts entities and the main topic from the given text using the llama.cpp server.

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
            client = ChatLlamaCppServer(
                model=self.model_alias,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            messages = [
                SystemMessage(
                    content="""You are an NER model that extracts entities and the topic from a text.
                    The output MUST be a valid JSON object strictly formatted as:
                    {"topic": "Technology", "entities": ["Elon Musk", "SpaceX", "Tesla", "Paris"]}.
                    Output ONLY the JSON object. No prose, no preamble, no markdown formatting."""
                ),
                HumanMessage(content=text),
            ]

            response = client.invoke(messages, stop=stop)
            self.logger.info("llama.cpp API call successful.")

            content = response.content
            if not content:
                raise RuntimeError("API response content is None")

            result = content if isinstance(content, str) else json.dumps(content)
            result = result.strip()

            # Sanitize markdown code fences if wrapped by the model
            if result.startswith("```"):
                result = result.strip("```json").strip("```").strip()

            self.logger.debug("Raw API response: %s", result)

            tokens = (
                response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if hasattr(response, "response_metadata")
                else 0
            )

            return json.loads(result), tokens

        except Exception as e:
            self.logger.exception("Error extracting topic and entities: %s", e)
            raise

    def find_similar_entities_globally(
        self, entities, max_tokens=1024, temperature=0.0, stop=None
    ):
        """
        Finds unified versions of entities by analyzing them in context using llama.cpp server.

        Args:
            entities (list): A list of entities to find unified versions for.

        Returns:
            tuple: (dict mapping unified names to originals, tokens_used)
        """
        self.logger.debug("Finding similar entities globally...")
        tokens = 0

        if not entities:
            return {}, 0

        try:
            client = ChatLlamaCppServer(
                model=self.model_alias,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            system_prompt = f"""Please normalize or unify the following list of entities:
                {json.dumps(entities)}

                You must output a single JSON object where the keys are the EXACT original entity names,
                and the values are the single unified version of that entity.
                If an entity has multiple valid representations, variations, synonyms, or acronyms, select the most common or widely recognized form.
                If an entity is already unified or does not require normalization, map it to itself.
                Do not include any extra information, preamble, notes, or markdown formatting. Output ONLY the JSON object.

                Example output format:
                {{"U.S.A.": "United States", "USA": "United States", "Apple Inc": "Apple", "Elon Musk": "Elon Musk"}}"""

            messages = [SystemMessage(content=system_prompt)]

            response = client.invoke(messages, stop=stop)

            response_content = response.content
            tokens = (
                response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if hasattr(response, "response_metadata")
                else 0
            )
            self.logger.debug(f"Response content: {response_content}")

            if not response_content:
                raise RuntimeError("API response content is None")

            result_text = (
                response_content
                if isinstance(response_content, str)
                else json.dumps(response_content)
            ).strip()

            # Sanitize markdown code fences if wrapped by the model
            if result_text.startswith("```"):
                result_text = result_text.strip("```json").strip("```").strip()

            unified_mapping = json.loads(result_text)

            entity_groups = defaultdict(list)
            for original_entity, unified_entity in unified_mapping.items():
                if original_entity in entities:
                    entity_groups[unified_entity].append(original_entity)

            # Failsafe: catch any entities the model omitted from the dictionary keys
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
