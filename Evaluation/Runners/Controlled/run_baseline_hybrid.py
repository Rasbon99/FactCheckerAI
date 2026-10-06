import os
import sqlite3
import time
import uuid
import re
import dotenv
from typing import List
from llamacpp_client import ChatLlamaCppServer, load_models, set_alias_map
from langchain_core.messages import HumanMessage
from log import Logger

# --- LangChain Imports ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

# --- Import Pipeline Components ---
from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_response_format_instructions
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer, Experiment
from Utils.embedding_handler import get_embedding_model

dotenv.load_dotenv("key.env", override=False)

model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")
model_port = int(os.getenv("LLM_MODEL_PORT", "8080"))

print(f"[Backend] Connecting to local llama.cpp server on port {model_port}...")
set_alias_map({model_alias: model_port})
load_models([model_alias])

MAX_CLAIMS_TO_TEST = 5

# Initialize llama.cpp configuration
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


def get_hybrid_verdict(claim_text, retrieved_evidence, response_format_instructions):
    """Asks the LLM to verify the claim using the Hybrid RAG retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 

    {response_format_instructions}

    EVIDENCE:
    {retrieved_evidence}

    CLAIM: {claim_text}
    """

    client = ChatLlamaCppServer(
        model=model_alias,
        temperature=0.0,
        max_tokens=200,
    )

    messages = [HumanMessage(content=prompt)]
    response = client.invoke(messages)

    content = response.content
    tokens = (
        response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        if hasattr(response, "response_metadata")
        else 0
    )

    return content, tokens


def run_hybrid_baseline():
    successful_runs = 0
    failed_runs = 0

    try:
        dataset_manager = DatasetManager()

        # Using the new metadata function
        metadata = dataset_manager.get_experiment_metadata(environment="controlled")
        active_dataset = metadata["dataset_name"]
        use_meta = metadata["use_metadata"]

        response_format_instructions = get_dataset_response_format_instructions(
            active_dataset
        )

        logger.info(
            f"Starting Baseline (HybridRAG Re-ranking) with {MAX_CLAIMS_TO_TEST} claims..."
        )
        logger.info(f"Environment: {metadata['environment']}")
        logger.info(f"Active Dataset: {active_dataset}")
        logger.info(f"Experiment Type: {metadata['experiment_type']}")
        logger.info(f"Using Metadata Super Query: {use_meta}")

        logger.info(
            "Loading Hugging Face Embeddings natively (This takes a few seconds)..."
        )

        embeddings = get_embedding_model()
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

        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_id = None
            claim_text = ""
            ground_truth = ""
            search_query = ""
            current_stage = "claim_setup"

            latency_retrieval = 0.0
            latency_generation = 0.0
            tokens_used = 0

            raw_sources = []
            query_result = None

            try:
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                search_query = claim_text
                if active_dataset == "AVERITEC" and use_meta:
                    search_query = dataset_manager.build_search_query(data)

                logger.info(
                    f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}"
                )
                if search_query != claim_text:
                    logger.info(f"Enriched Search Query: {search_query}")
                logger.info(f"Ground Truth: {ground_truth}")

                claim_id = str(uuid.uuid4())

                Claim(
                    text=claim_text,
                    title="[Hybrid] " + claim_text[:30] + "...",
                    claim_id=claim_id,
                )

                # --- THE RETRIEVAL STEP ---
                current_stage = "retrieval"

                logger.info("Extracting and Re-ranking...")
                t0 = time.time()
                best_docs = []

                if active_dataset == "FEVER":
                    if hybrid_rag_retriever is None:
                        raise RuntimeError(
                            "FEVER Retriever was not properly initialized."
                        )
                    best_docs = hybrid_rag_retriever.invoke(search_query)

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

                        # 3. STAGE 2: Heavy Semantic Re-ranking (Filter 50 down to 2)
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=embeddings_filter,
                            base_retriever=bm25_retriever,
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
                current_stage = "generation"

                t0 = time.time()
                query_result, tokens_used = get_hybrid_verdict(
                    claim_text, combined_evidence, response_format_instructions
                )
                latency_generation = time.time() - t0

                # --- 3. Verdict Parsing ---
                current_stage = "parsing"

                error_type = None
                error_message = None
                error_stage = None

                try:
                    if not query_result or not query_result.strip():
                        predicted_label = "Error"
                        error_type = "EmptyLLMResponse"
                        error_message = "The LLM returned an empty response."
                        error_stage = "parsing"
                        query_result = "The LLM failed to generate a response."

                    elif "VERDICT:" in query_result and "REASONING:" in query_result:
                        predicted_label = (
                            query_result.split("REASONING:")[0]
                            .replace("VERDICT:", "")
                            .strip()
                        )

                    else:
                        predicted_label = "Error"
                        error_type = "UnstructuredResponse"
                        error_message = "The LLM response did not contain both VERDICT and REASONING."
                        error_stage = "parsing"

                except Exception as e:
                    logger.exception(f"Error parsing verdict: {e}")
                    predicted_label = "Error"
                    error_type = type(e).__name__
                    error_message = str(e)
                    error_stage = "parsing"

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
                    tokens={
                        "preprocessor": 0,
                        "retrieval": 0,
                        "generation": tokens_used,
                    },
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
                    use_metadata=use_meta,
                    error_details=(
                        f"Stage: {error_stage} | "
                        f"Type: {error_type} | "
                        f"Message: {error_message}"
                        if error_type is not None
                        else None
                    ),
                )

                if predicted_label == "Error":
                    failed_runs += 1
                else:
                    successful_runs += 1

            except Exception as e:
                logger.exception(f"Error processing claim {line_number + 1}: {e}")

                try:
                    if claim_id is not None:
                        Experiment(
                            claim_id=claim_id,
                            predicted_label="Error",
                            ground_truth=ground_truth,
                            latencies={
                                "preprocessor": 0.0,
                                "retrieval": latency_retrieval,
                                "generation": latency_generation,
                            },
                            tokens={
                                "preprocessor": 0,
                                "retrieval": 0,
                                "generation": tokens_used,
                            },
                            calls={
                                "preprocessor": 0,
                                "retrieval": 0,
                                "generation": 1 if query_result is not None else 0,
                            },
                            evidence_data={
                                "claim_text": claim_text,
                                "search_query": search_query,
                                "raw_sources": raw_sources,
                                "query_result": query_result,
                            },
                            system_type="HybridRAG",
                            environment=metadata["environment"],
                            dataset_name=metadata["dataset_name"],
                            experiment_type=metadata["experiment_type"],
                            use_metadata=use_meta,
                            error_details=(
                                f"Stage: {current_stage} | "
                                f"Type: {type(e).__name__} | "
                                f"Message: {str(e)}"
                            ),
                        )
                except Exception:
                    logger.exception(
                        f"Failed to save failed experiment for claim {line_number + 1}"
                    )

                failed_runs += 1
                continue

    except Exception as e:
        logger.exception(f"Fatal error during experiment: {e}")

    logger.info("=" * 20)
    logger.info("HYBRID-RAG BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_hybrid_baseline()
