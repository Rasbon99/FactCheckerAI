import os
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
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

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
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"
client = Groq(api_key=GROQ_API_KEY)
logger = Logger("Hybrid-Baseline").get_logger()


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
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("Controlled")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline 3 (Hybrid-RAG Re-ranking) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
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
        # Snap the SQLite search and the Embedding filter together
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

            # --- CONDITIONAL METADATA QUERY ---
            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())
            tracker = ExperimentTracker(
                claim_id=claim_id,
                ground_truth=ground_truth,
                system_type="Baseline-Hybrid",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[Hybrid] " + claim_text[:30] + "...",
                summary="Tested with Two-Stage Retrieval (BM25 -> Dense Embeddings).",
                claim_id=claim_id,
            )

            # --- THE RETRIEVAL STEP ---
            logger.info("Extracting and Re-ranking with Ollama...")

            def run_retrieval():
                if active_dataset == "FEVER":
                    if hybrid_rag_retriever is None:
                        raise RuntimeError(
                            "FEVER Retriever was not properly initialized."
                        )
                    # Use the search_query (which is identical to claim_text for FEVER)
                    return hybrid_rag_retriever.invoke(search_query)

                elif active_dataset == "AVERITEC":
                    if averitec_retriever is None:
                        raise RuntimeError(
                            "AVeriTeC Retriever was not properly initialized."
                        )

                    claim_id_internal = data.get("internal_id")
                    sentences = averitec_retriever.get_evidence_for_claim(
                        claim_id_internal
                    )

                    # INJECT NOISE (If running the Noisy robustness test)
                    if "noisy_ids" in data and sentences is not None:
                        for n_id in data["noisy_ids"]:
                            noisy_sentences = averitec_retriever.get_evidence_for_claim(
                                n_id
                            )
                            if noisy_sentences:
                                sentences.extend(noisy_sentences)

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

                    # Execute the two-stage pipeline using the ENRICHED query
                    return compression_retriever.invoke(search_query)

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
                # Strictly pass the original claim_text, not the search_query
                res_text, toks = get_hybrid_verdict(
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

            logger.info(f"Hybrid Verdict: {predicted_label}")

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
    logger.info("HYBRID-RAG BASELINE COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_hybrid_baseline()
