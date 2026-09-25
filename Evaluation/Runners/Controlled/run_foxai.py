import os
import sqlite3
import time
import uuid
import dotenv
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_prompt_instructions
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from GraphRAG.rag_pipeline import RAG_Pipeline
from Database.data_entities import Claim, Answer, Experiment

dotenv.load_dotenv("key.env", override=False)

MAX_CLAIMS_TO_TEST = 5
logger = Logger("Fox-AI-Controlled").get_logger()


def extract_perfect_evidence(evidence_data, wiki_cursor):
    """
    Parses the FEVER evidence JSON, queries the SQLite DB, and extracts the exact text.
    """
    extracted_text = ""

    for evidence_set in evidence_data:
        for ev in evidence_set:
            page_id = ev[2]
            sentence_num = str(ev[3])

            if page_id is None:
                continue

            wiki_cursor.execute(
                "SELECT lines FROM wiki_articles WHERE page_id = ?", (page_id,)
            )
            result = wiki_cursor.fetchone()

            if result:
                raw_lines = result[0]
                sentences = raw_lines.split("\n")
                for sentence in sentences:
                    parts = sentence.split("\t")
                    if parts[0] == sentence_num and len(parts) > 1:
                        extracted_text += parts[1] + " "
                        break

    return extracted_text.strip()


def run_controlled_experiment():
    dataset_manager = DatasetManager()

    metadata = dataset_manager.get_experiment_metadata(environment="controlled")
    active_dataset = metadata["dataset_name"]
    use_meta = metadata["use_metadata"]

    prompt_instructions = get_dataset_prompt_instructions(active_dataset)

    logger.info(
        f"Starting Controlled Experiment (FoxAI) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Metadata Super Query: {use_meta}")

    wiki_conn = None
    wiki_cursor = None
    averitec_retriever = None

    if active_dataset == "FEVER":
        wiki_db_path = os.getenv(
            "FEVER_WIKIPEDIA_DB_PATH", "Datasets/FEVER/fever_wiki.db"
        )
        wiki_conn = sqlite3.connect(wiki_db_path)
        wiki_cursor = wiki_conn.cursor()
    elif active_dataset == "AVERITEC":
        averitec_retriever = AVeriTeCKnowledgeRetriever()

    preprocessor = Preprocessing_Pipeline()
    rag = RAG_Pipeline()

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            search_query = claim_text
            if active_dataset == "AVERITEC" and use_meta:
                search_query = dataset_manager.build_search_query(data)

            nei_label = (
                "NOT ENOUGH INFO"
                if active_dataset == "FEVER"
                else "Not Enough Evidence"
            )

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")
            logger.info(f"Ground Truth: {ground_truth}")

            # --- Extract Evidence Dynamically ---
            perfect_evidence = ""
            if active_dataset == "FEVER":
                evidence_data = data.get("evidence", [])
                logger.info(f"Raw Evidence Array from FEVER: {evidence_data}")
                perfect_evidence = extract_perfect_evidence(evidence_data, wiki_cursor)

            elif active_dataset == "AVERITEC" and averitec_retriever is not None:
                claim_id_internal = data.get("internal_id")
                all_sentences = averitec_retriever.get_evidence_for_claim(
                    claim_id_internal
                )

                if "noisy_ids" in data and all_sentences is not None:
                    for n_id in data["noisy_ids"]:
                        noisy_sentences = averitec_retriever.get_evidence_for_claim(
                            n_id
                        )
                        if noisy_sentences:
                            all_sentences.extend(noisy_sentences)

                if all_sentences:
                    from rank_bm25 import BM25Okapi

                    tokenized_corpus = [s.lower().split() for s in all_sentences]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = search_query.lower().split()

                    top_sentences = bm25.get_top_n(tokenized_query, all_sentences, n=30)
                    perfect_evidence = " ".join(top_sentences)

            if not perfect_evidence and ground_truth != nei_label:
                logger.warning(
                    "Could not find evidence in knowledge base for this claim."
                )

            logger.info(f"Perfect Evidence Retrieved: {perfect_evidence[:100]}...")

            latencies = {}
            tokens = {}
            calls = {}

            # --- THE SHORT-CIRCUIT ---
            if not perfect_evidence:
                logger.info(f"No evidence available. Short-circuiting to {nei_label}.")
                claim_id = str(uuid.uuid4())

                Claim(
                    text=claim_text,
                    title="[NEI] " + claim_text[:30] + "...",
                    summary="Short-circuited due to lack of evidence.",
                    claim_id=claim_id,
                )

                predicted_label = nei_label
                query_result = f"VERDICT: {nei_label}\nREASONING: No evidence provided by the dataset."

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                Experiment(
                    claim_id=claim_id,
                    predicted_label=predicted_label,
                    ground_truth=ground_truth,
                    latencies={
                        "preprocessor": 0.0,
                        "retrieval": 0.0,
                        "generation": 0.0,
                    },
                    tokens={"preprocessor": 0, "retrieval": 0, "generation": 0},
                    calls={"preprocessor": 0, "retrieval": 0, "generation": 0},
                    evidence_data={
                        "claim_text": claim_text,
                        "raw_sources": [],
                        "query_result": query_result,
                    },
                    system_type="FoxAI-GraphRAG",
                    environment=metadata["environment"],
                    dataset_name=metadata["dataset_name"],
                    experiment_type=metadata["experiment_type"],
                    use_metadata=use_meta,
                )

                successful_runs += 1
                continue

            # --- NORMAL PIPELINE ---
            claim_id = str(uuid.uuid4())

            # 1. Preprocessing
            t0 = time.time()
            prep_data, prep_claim_metrics = preprocessor.run_claim_pipe(claim_text)
            claim_title, claim_summary = prep_data

            # SAFETY FALLBACK: If LLM fails to summarize, use the raw claim text
            if not claim_title:
                claim_title = f"!g {claim_text[:50]}..."
            if not claim_summary:
                claim_summary = claim_text

            latencies["preprocessor"] = time.time() - t0
            tokens["preprocessor"] = prep_claim_metrics.get("total", 0)
            calls["preprocessor"] = prep_claim_metrics.get("calls", 0)

            claim = Claim(claim_text, claim_title, claim_summary, claim_id=claim_id)

            # 2. Retrieval (Mock Scraper)
            t0 = time.time()
            mock_srcs = [
                {
                    "title": f"{active_dataset} Perfect Evidence",
                    "url": "Local_DB",
                    "site": "Dataset",
                    "body": perfect_evidence,
                }
            ]
            preprocessed_sources, prep_metrics = preprocessor.run_sources_pipe(
                mock_srcs
            )
            latencies["retrieval"] = time.time() - t0
            tokens["retrieval"] = prep_metrics.get("total", 0)
            calls["retrieval"] = prep_metrics.get("calls", 0)

            claim.add_sources(preprocessed_sources)

            # Safety Check: Entities
            has_entities = False
            for src in preprocessed_sources:
                entities = src.get("entities")
                if entities is not None and entities != "[]" and len(entities) > 0:
                    has_entities = True
                    break

            if not has_entities:
                logger.info(
                    "NER extracted 0 entities. Short-circuiting to prevent Neo4j crash."
                )
                predicted_label = "Error: No Entities"
                query_result = f"VERDICT: {nei_label}\nREASONING: Evidence was provided, but the NER model failed to extract any entities to build a graph."
                Answer(claim_id=claim.id, answer=query_result, graphs_folder=None)

                Experiment(
                    claim_id=claim_id,
                    predicted_label=predicted_label,
                    ground_truth=ground_truth,
                    latencies=latencies,
                    tokens=tokens,
                    calls=calls,
                    evidence_data={
                        "claim_text": claim_text,
                        "raw_sources": mock_srcs,
                        "query_result": query_result,
                    },
                    system_type="FoxAI-GraphRAG",
                    environment=metadata["environment"],
                    dataset_name=metadata["dataset_name"],
                    experiment_type=metadata["experiment_type"],
                    use_metadata=use_meta,
                )
                successful_runs += 1
                continue

            # 3. GraphRAG Generation
            t0 = time.time()
            query_result, graphs_folder, t_usage = rag.run_pipeline(
                preprocessed_sources, claim.text, claim.id, prompt_instructions
            )
            latencies["generation"] = time.time() - t0
            tokens["generation"] = t_usage.get("llm_total", t_usage.get("total", 0))
            calls["generation"] = t_usage.get("llm_calls", t_usage.get("calls", 0))

            # 4. Verdict Parsing
            try:
                if isinstance(query_result, list):
                    query_result = "\n".join(
                        item if isinstance(item, str) else str(item)
                        for item in query_result
                    )
                elif query_result is not None and not isinstance(query_result, str):
                    query_result = str(query_result)

                if not query_result or not query_result.strip():
                    predicted_label = "Error: Empty LLM Response"
                    query_result = "The LLM failed to generate a response."
                elif "VERDICT:" in query_result:
                    predicted_label = (
                        query_result.split("REASONING:")[0]
                        .replace("VERDICT:", "")
                        .strip()
                    )
                else:
                    predicted_label = "Error: Unstructured Response"
            except Exception:
                predicted_label = "Parsing Error"

            logger.info(f"FoxAI Verdict: {predicted_label}")

            Answer(claim_id=claim.id, answer=query_result, graphs_folder=graphs_folder)

            # 5. Log Experiment Metrics
            Experiment(
                claim_id=claim_id,
                predicted_label=predicted_label,
                ground_truth=ground_truth,
                latencies=latencies,
                tokens=tokens,
                calls=calls,
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": mock_srcs,
                    "query_result": query_result,
                },
                system_type="FoxAI-GraphRAG",
                environment=metadata["environment"],
                dataset_name=metadata["dataset_name"],
                experiment_type=metadata["experiment_type"],
                use_metadata=use_meta,
            )

            successful_runs += 1

            logger.info("Sleeping for 5 seconds before next claim...")
            time.sleep(5)

    except FileNotFoundError as e:
        logger.error(f"Could not find dataset files. {e}")
    finally:
        if wiki_conn:
            wiki_conn.close()

    logger.info("=" * 20)
    logger.info("CONTROLLED EXPERIMENT COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_controlled_experiment()
