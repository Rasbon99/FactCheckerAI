import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings


class NomicHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Custom wrapper for Nomic embeddings to automatically append mandatory
    task prefixes ('search_document: ' and 'search_query: ') to prevent
    out-of-distribution retrieval degradation.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Add the document prefix to every chunk of text being saved to the database
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        # Add the query prefix to the user's claim/question
        prefixed_text = f"search_query: {text}"
        return super().embed_query(prefixed_text)


def get_embedding_model(model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
    """
    Utility function to instantly initialize the Nomic embedding model
    with all required configurations (remote code, normalization).
    """
    return NomicHuggingFaceEmbeddings(
        model=model_name,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
