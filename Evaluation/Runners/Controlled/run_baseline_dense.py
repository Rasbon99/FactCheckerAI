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

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Database.data_entities import Claim, Answer

dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
WIKI_DB_PATH = os.getenv("FEVER_WIKIPEDIA_DB_PATH", "Datasets/fever_wiki.db")
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger(__name__).get_logger()


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


def get_dense_verdict(claim_text, retrieved_evidence):
    """Asks the LLM to verify the claim using the Dense retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
Verify the following claim using ONLY the provided evidence. 
If the evidence does not contain enough information to make a definitive decision, answer NOT ENOUGH INFO.

You must format your response EXACTLY like this:
VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
REASONING: [Your brief explanation citing the provided evidence]

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
    logger.info(
        f"Starting Baseline 3 (Dense-RAG Re-ranking) with {MAX_CLAIMS_TO_TEST} claims..."
    )

    # ---------------------------------------------------------
    # 2. THE LANGCHAIN TWO-STAGE PIPELINE (STAGE 2: EMBEDDINGS)
    # ---------------------------------------------------------
    logger.info("Loading Ollama Embeddings (This takes a few seconds)...")
    # Note: If you ever want to use OpenAI, swap this to OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the filter: it will embed the 50 docs, compare to the claim, and keep the top 2
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, k=2)

    # Snap the SQLite search and the Embedding filter together
    dense_rag_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=SQLiteFTS5Retriever(db_path=WIKI_DB_PATH),
    )
    # ---------------------------------------------------------

    successful_runs = 0

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                logger.info(
                    f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}"
                )

                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id,
                    ground_truth=ground_truth,
                    system_type="Baseline-Dense",  # 🚀 LABELED AS DENSE
                    dataset_setting="FEVER-Controlled",
                )

                Claim(
                    text=claim_text,
                    title="[Dense] " + claim_text[:30] + "...",
                    summary="Tested with Two-Stage Dense Retrieval.",
                    claim_id=claim_id,
                )

                # --- THE RETRIEVAL STEP (LangChain Invocation) ---
                logger.info("Searching SQLite & Re-ranking with Ollama...")

                def run_retrieval():
                    # This single line runs the SQLite query AND the Ollama embedding re-ranking!
                    return dense_rag_retriever.invoke(claim_text)

                # Run the stopwatch!
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
                    combined_evidence = "No relevant Wikipedia articles found."

                # --- THE GENERATION STEP ---
                def run_llm():
                    res_text, toks = get_dense_verdict(claim_text, combined_evidence)
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

    logger.info("=" * 40)
    logger.info("DENSE-RAG BASELINE COMPLETE!")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_dense_baseline()
