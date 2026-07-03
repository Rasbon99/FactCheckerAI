import json
import os
import dotenv
from llamacpp_client import ChatLlamaCppServer
from langchain_core.messages import SystemMessage, HumanMessage
from collections import defaultdict

from log import Logger


class NER:
    def __init__(self, env_file="key.env"):
        """
        Initializes the NER class with a specific model and configures the llama.cpp client.

        Args:
            env_file (str, optional): The path to the environment file containing configuration. Default is "key.env".
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")

    def extract_entities_and_topic(
        self, text, max_tokens=1024, temperature=0.5, stop=None
    ):
        """
        Extracts entities and the main topic from the given text using the llama.cpp server.

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
            client = ChatLlamaCppServer(
                model=self.model_alias, temperature=temperature, max_tokens=max_tokens
            )

            messages = [
                SystemMessage(
                    content="""you are an NER model that extracts entities and the topic from a text.\n 
                    The output must be strictly formatted as: {\"topic\": \"Technology\", \"entities\": [\"Elon Musk\", \"SpaceX\", \"Tesla\", \"Paris\"]}. No prose, no markdown formatting."""
                ),
                HumanMessage(content=text),
            ]

            response = client.invoke(messages, stop=stop)

            self.logger.info("llama.cpp API call successful.")
            content = response.content
            if content is None:
                self.logger.error("API response content is None")
                return None, 0
            result = content if isinstance(content, str) else json.dumps(content)
            result = result.strip()

            # Sanitize raw markdown block wrapper if injected by the model
            if result.startswith("```"):
                result = result.strip("```json").strip("```").strip()

            self.logger.debug("Raw API response: %s", result)

            tokens = (
                response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if hasattr(response, "response_metadata")
                else 0
            )

            return json.loads(result), tokens
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error("Error extracting topic and entities: %s", e)
            return None, 0

    def find_similar_entities_globally(
        self, entities, max_tokens=1024, temperature=0.0, stop=None
    ):
        """
        Finds unified versions of entities by analyzing them in context using llama.cpp server.

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

            client = ChatLlamaCppServer(
                model=self.model_alias, temperature=temperature, max_tokens=max_tokens
            )

            messages = [
                SystemMessage(
                    content=f"""Please normalize or unify the following entities: {input_entities}. 
                                    For each entity, return a single unified version. 
                                    If an entity has multiple valid representations, variations, synonyms, or acronyms, select the most common or widely recognized form. 
                                    Ensure the unified versions are returned in the same order as the input, separated by commas, and the total number of unified entities matches the number of input entities. 
                                    If any entity is already unified or does not require normalization, return it as is. 
                                    Do not include any extra information, notes, or context.
                                    Example: 
                                        Input: ['United States', 'USA', 'US', 'U.S.'] Output: United States, United States, United States, United States"""
                )
            ]

            response = client.invoke(messages, stop=stop)

            # Extraction from response
            response_content = response.content
            tokens = (
                response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if hasattr(response, "response_metadata")
                else 0
            )
            self.logger.debug(f"Response content: {response_content}")

            # Handling the response carefully
            if response_content is None:
                self.logger.error("API response content is None")
                return {entity: [entity] for entity in entities}, 0

            # Convert to string if needed
            response_str = (
                response_content
                if isinstance(response_content, str)
                else json.dumps(response_content)
            )

            # Strip out accidental list bracket formatting if model adds them
            clean_content = (
                response_str.strip()
                .strip("[")
                .strip("]")
                .replace("'", "")
                .replace('"', "")
            )
            unified_entities_list = [ue.strip() for ue in clean_content.split(",")]

            # Ensure the number of unified entities matches the input entities count
            if len(unified_entities_list) != len(entities):
                raise ValueError(
                    "The number of unified entities does not match the number of input entities."
                )

            # Map original entities to their unified versions
            unified_mapping = {
                entities[i]: unified_entities_list[i] for i in range(len(entities))
            }

            # Group entities by their unified version
            entity_groups = defaultdict(list)
            for entity, unified in unified_mapping.items():
                entity_groups[unified].append(entity)

            self.logger.debug(f"Grouped entities globally: {dict(entity_groups)}")

            return entity_groups, tokens

        except Exception as e:
            self.logger.error(f"Error in global entity similarity analysis: {e}")
            # Fallback: return each entity as its own group
            return {entity: [entity] for entity in entities}, 0

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

        # Merge the entities across the sources
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
            source["entities"] = list(set(updated_entities))  # Remove duplicates

        self.logger.info("Entities merged and sources updated successfully.")
        return sources, tokens
