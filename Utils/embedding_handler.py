import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv("key.env", override=False)


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
        if self.is_nomic:
            texts = [f"search_document: {text}" for text in texts]

        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.is_nomic:
            text = f"search_query: {text}"

        return super().embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR")

    if not model_name:
        raise RuntimeError(
            "EMBEDDING_MODEL_NAME is not set in key.env"
        )

    if not cache_dir:
        raise RuntimeError(
            "EMBEDDING_CACHE_DIR is not set in key.env"
        )

    cache_path = Path(cache_dir)

    if not cache_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        cache_path = project_root / cache_path

    cache_path.mkdir(parents=True, exist_ok=True)

    model_kwargs = {}

    if "nomic" in model_name.lower():
        model_kwargs["trust_remote_code"] = True

    return UniversalHuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(cache_path),
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )