from typing import ClassVar, List

from langchain_huggingface import HuggingFaceEmbeddings


class BGEHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    BGE embedding wrapper.

    Documents are embedded normally.
    Retrieval queries use BGE's retrieval instruction.
    """

    QUERY_INSTRUCTION: ClassVar[str] = (
        "Represent this sentence for searching relevant passages: "
    )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        query = f"{self.QUERY_INSTRUCTION}{text}"
        return super().embed_query(query)


def get_embedding_model(
    model_name: str = "BAAI/bge-base-en-v1.5",
):
    return BGEHuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )