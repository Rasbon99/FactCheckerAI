import os
import sqlite3
import time
import uuid
import re
import dotenv
from typing import List
from groq import Groq
from log import Logger

# --- LangChain Imports ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# --- Import Pipeline Components ---
from Evaluation.Utils.dataset_manager import DatasetManager
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer, Experiment

dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"

client = Groq(api_key=GROQ_API_KEY)
logger = Logger("HybridRAG-Controlled").get_logger()


# ====================================================================
# 1. THE CUSTOM SQLITE RETRIEVER (STAGE 1: BM25 / FAST & CHEAP)
# ====================================================================
class SQLiteFTS5Retriever(BaseRetriever):
    db_path: str
    top_k: int = 50  # Grab 50 docs quickly using keyword matching

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        clean_query = re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()

        cursor.execute(
            """
            SELECT page_id, lines FROM wiki_fts 
            WHERE wiki_fts MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """,
            (clean_query, self.top_k),
        )

        rows = cursor.fetchall()
        conn.close()

        # Convert SQLite rows into LangChain Document objects
        return [
            Document(
                page_content=f"--- Article: {row[0]} ---\n{row[1][:1200]}...",
                metadata={"source": row[0]},
            )
            for row in rows
        ]


# ====================================================================


def get_hybrid_verdict(claim_text, retrieved_evidence, prompt_instructions, nei_label):
    """Asks the LLM to verify the claim using the Hybrid RAG retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer exactly: {nei_label}.

    {prompt_instructions}

    EVIDENCE:
    {retrieved_evidence}

    CLAIM: {claim_text}
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=200,
    )

    return response.choices[0].message.content, (
        response.usage.total_tokens if response.usage else 0
    )


def run_hybrid_baseline():
    dataset_manager = DatasetManager()

    # Using the new metadata function
    metadata = dataset_manager.get_experiment_metadata(environment="controlled")
    active_dataset = metadata["dataset_name"]

    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline (HybridRAG Re-ranking) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")

    # ---------------------------------------------------------
    # 2. CONFIGURE THE RETRIEVAL PIPELINE BASED ON DATASET
    # ---------------------------------------------------------
    logger.info(
        "Loading Hugging Face Embeddings natively (This takes a few seconds)..."
    )
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, k=2)

    hybrid_rag_retriever = None
    averitec_retriever = None

    if active_dataset == "FEVER":
        wiki_db_path = os.getenv(
            "FEVER_WIKIPEDIA_DB_PATH", "Datasets/FEVER/fever_wiki.db"
        )
        hybrid_rag_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=SQLiteFTS5Retriever(db_path=wiki_db_path),
        )
    elif active_dataset == "AVERITEC":
        averitec_retriever = AVeriTeCKnowledgeRetriever()
    # ---------------------------------------------------------

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[Hybrid] " + claim_text[:30] + "...",
                summary="Tested with Two-Stage Retrieval (BM25 -> Dense Embeddings).",
                claim_id=claim_id,
            )

            # --- THE RETRIEVAL STEP ---
            t0 = time.time()
            best_docs = []

            if active_dataset == "FEVER":
                if hybrid_rag_retriever is None:
                    raise RuntimeError("FEVER Retriever was not properly initialized.")
                best_docs = hybrid_rag_retriever.invoke(search_query)

            elif active_dataset == "AVERITEC":
                if averitec_retriever is None:
                    raise RuntimeError(
                        "AVeriTeC Retriever was not properly initialized."
                    )

                claim_id_internal = data.get("internal_id")
                sentences = averitec_retriever.get_evidence_for_claim(claim_id_internal)

                # INJECT NOISE (If running the Noisy robustness test)
                if "noisy_ids" in data and sentences is not None:
                    for n_id in data["noisy_ids"]:
                        noisy_sentences = averitec_retriever.get_evidence_for_claim(
                            n_id
                        )
                        if noisy_sentences:
                            sentences.extend(noisy_sentences)

                if sentences:
                    docs = [
                        Document(
                            page_content=s,
                            metadata={
                                "source": f"AVeriTeC Store ID: {claim_id_internal}"
                            },
                        )
                        for s in sentences
                    ]
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 50

                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=embeddings_filter, base_retriever=bm25_retriever
                    )
                    best_docs = compression_retriever.invoke(search_query)

            latency_retrieval = time.time() - t0

            combined_evidence = ""
            raw_sources = []

            for doc in best_docs:
                combined_evidence += f"\n{doc.page_content}\n"
                raw_sources.append(
                    {
                        "title": doc.metadata.get("source", "Unknown"),
                        "snippet": doc.page_content,
                    }
                )

            if not combined_evidence:
                combined_evidence = (
                    f"No relevant evidence found in the {active_dataset} database."
                )

            # --- THE GENERATION STEP ---
            t0 = time.time()
            query_result, tokens_used = get_hybrid_verdict(
                claim_text, combined_evidence, prompt_instructions, nei_label
            )
            latency_generation = time.time() - t0

            # Verdict Parsing
            try:
                if query_result and "VERDICT:" in query_result:
                    predicted_label = (
                        query_result.split("REASONING:")[0]
                        .replace("VERDICT:", "")
                        .strip()
                    )
                else:
                    predicted_label = "Error: Unstructured Response"
            except Exception:
                predicted_label = "Parsing Error"

            logger.info(f"Hybrid Verdict: {predicted_label}")

            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # --- LOG TO EXPERIMENTS DATABASE ---
            Experiment(
                claim_id=claim_id,
                predicted_label=predicted_label,
                ground_truth=ground_truth,
                latencies={
                    "preprocessor": 0.0,
                    "retrieval": latency_retrieval,
                    "generation": latency_generation,
                },
                tokens={"preprocessor": 0, "retrieval": 0, "generation": tokens_used},
                calls={"preprocessor": 0, "retrieval": 0, "generation": 1},
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": raw_sources,
                    "query_result": query_result,
                },
                system_type="HybridRAG",
                environment=metadata["environment"],
                dataset_name=metadata["dataset_name"],
                experiment_type=metadata["experiment_type"],
            )

            successful_runs += 1

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("HYBRID-RAG BASELINE COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_hybrid_baseline()
