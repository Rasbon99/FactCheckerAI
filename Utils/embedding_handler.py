import os
from functools import lru_cache
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings


class UniversalHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Custom wrapper for Hugging Face embeddings that dynamically applies
    mandatory task prefixes based on the active model's architecture.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def is_nomic(self) -> bool:
        return "nomic" in self.model_name.lower()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Apply the document prefix ONLY if the model is Nomic
        if self.is_nomic:
            texts = [f"search_document: {text}" for text in texts]

        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # Apply the query prefix ONLY if the model is Nomic
        if self.is_nomic:
            text = f"search_query: {text}"

        return super().embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Utility function to instantly initialize and cache the embedding model
    in memory. It dynamically reads the active model from the environment.
    """
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")

    return UniversalHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
