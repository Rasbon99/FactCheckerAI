import os
import json
import sqlite3
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
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Evaluation.Utils.dataset_manager import DatasetManager
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer

dotenv.load_dotenv("key.env", override=True)

# Configuration
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger("Dense-Baseline").get_logger()


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


def get_dense_verdict(claim_text, retrieved_evidence, prompt_instructions, nei_label):
    """Asks the LLM to verify the claim using the Dense retrieved text."""
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


def run_dense_baseline():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("Controlled")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline 3 (Dense-RAG Re-ranking) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")

    # ---------------------------------------------------------
    # 2. CONFIGURE THE RETRIEVAL PIPELINE BASED ON DATASET
    # ---------------------------------------------------------
    logger.info("Loading Ollama Embeddings (This takes a few seconds)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, k=2)

    dense_rag_retriever = None
    averitec_retriever = None

    if active_dataset == "FEVER":
        wiki_db_path = os.getenv(
            "FEVER_WIKIPEDIA_DB_PATH", "Datasets/FEVER/fever_wiki.db"
        )
        # Snap the SQLite search and the Embedding filter together
        dense_rag_retriever = ContextualCompressionRetriever(
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

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")

            claim_id = str(uuid.uuid4())
            tracker = ExperimentTracker(
                claim_id=claim_id,
                ground_truth=ground_truth,
                system_type="Baseline-Dense",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[Dense] " + claim_text[:30] + "...",
                summary="Tested with Two-Stage Retrieval (BM25 -> Dense).",
                claim_id=claim_id,
            )

            # --- THE RETRIEVAL STEP ---
            logger.info("Extracting and Re-ranking with Ollama...")

            def run_retrieval():
                if active_dataset == "FEVER":
                    if dense_rag_retriever is None:
                        raise RuntimeError(
                            "FEVER Retriever was not properly initialized."
                        )
                    # Let LangChain handle the full SQLite -> Embedding pipeline
                    return dense_rag_retriever.invoke(claim_text)

                elif active_dataset == "AVERITEC":
                    if averitec_retriever is None:
                        raise RuntimeError(
                            "AVeriTeC Retriever was not properly initialized."
                        )

                    claim_id_internal = data.get("internal_id")
                    sentences = averitec_retriever.get_evidence_for_claim(
                        claim_id_internal
                    )

                    if not sentences:
                        return []

                    # 1. Wrap strings into LangChain Documents
                    docs = [
                        Document(
                            page_content=s,
                            metadata={
                                "source": f"AVeriTeC Store ID: {claim_id_internal}"
                            },
                        )
                        for s in sentences
                    ]

                    # 2. STAGE 1: Fast BM25 Math (Perfectly matches FEVER's top_k=50)
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 50

                    # 3. STAGE 2: Heavy Semantic Re-ranking (Filter 50 down to 2 using Ollama)
                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=embeddings_filter, base_retriever=bm25_retriever
                    )

                    # Execute the two-stage pipeline
                    return compression_retriever.invoke(claim_text)

            best_docs = tracker.run_stage("retrieval", run_retrieval)

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
            def run_llm():
                res_text, toks = get_dense_verdict(
                    claim_text, combined_evidence, prompt_instructions, nei_label
                )
                return (res_text, None), {"total": toks, "calls": 1}

            query_result = tracker.run_stage("generation", run_llm)

            if isinstance(query_result, tuple):
                query_result = query_result[0]

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

            logger.info(f"Dense Verdict: {predicted_label}")

            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # Log to DB
            tracker.finalize(
                predicted_label,
                {
                    "claim_text": claim_text,
                    "raw_sources": raw_sources,
                    "query_result": query_result,
                },
            )
            successful_runs += 1

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("DENSE-RAG BASELINE COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_dense_baseline()
