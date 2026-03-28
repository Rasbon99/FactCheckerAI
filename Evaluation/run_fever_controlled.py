import os
import json
import sqlite3
import time
import uuid
import dotenv

# --- Import Pipeline Components ---
from Evaluation.experiment_tracker import ExperimentTracker
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from GraphRAG.rag_pipeline import RAG_Pipeline
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
WIKI_DB_PATH = os.getenv("FEVER_WIKIPEDIA_DB_PATH", "Datasets/fever_wiki.db")
MAX_CLAIMS_TO_TEST = 100


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
    print(
        f"Starting Controlled Experiment A (Perfect Evidence) with {MAX_CLAIMS_TO_TEST} claims..."
    )

    # Connect to the Wikipedia DB (Read Only)
    wiki_conn = sqlite3.connect(WIKI_DB_PATH)
    wiki_cursor = wiki_conn.cursor()

    # Initialize the heavy pipeline components once (so we don't reload models every loop)
    preprocessor = Preprocessing_Pipeline()
    rag = RAG_Pipeline()

    successful_runs = 0

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")
                evidence_data = data.get("evidence", [])

                print(f"\n[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Raw Evidence Array from FEVER: {evidence_data}")

                # Extract the perfect evidence from the local DB!
                perfect_evidence = extract_perfect_evidence(evidence_data, wiki_cursor)

                if not perfect_evidence and ground_truth != "NOT ENOUGH INFO":
                    print("⚠️ Warning: Could not find evidence in DB for this claim.")

                print(f"Perfect Evidence Retrieved: {perfect_evidence[:100]}...")

                # --- THE SHORT-CIRCUIT ---
                # If there is no evidence, skip the heavy AI processing entirely!
                if not perfect_evidence:
                    print(
                        "⏩ No evidence available. Short-circuiting to NOT ENOUGH INFO."
                    )
                    claim_id = str(uuid.uuid4())
                    tracker = ExperimentTracker(
                        claim_id=claim_id, ground_truth=ground_truth
                    )

                    # 1. Create a Claim so it shows up in the UI sidebar
                    Claim(
                        text=claim_text,
                        title="[NEI] " + claim_text[:30] + "...",
                        summary="Short-circuited due to lack of FEVER evidence.",
                        claim_id=claim_id,
                    )

                    predicted_label = "NOT ENOUGH INFO"
                    query_result = "VERDICT: NOT ENOUGH INFO\nREASONING: No evidence provided by the FEVER dataset."

                    # 2. Create the Answer so the UI has text to display
                    Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                    tracker.finalize(
                        predicted_label,
                        {
                            "claim_text": claim_text,
                            "raw_sources": [],
                            "query_result": query_result,
                        },
                    )

                    successful_runs += 1
                    continue  # Jump immediately to the next claim in the loop!

                # --- NORMAL PIPELINE (Only runs if we have actual evidence) ---
                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id, ground_truth=ground_truth
                )

                # --- 1. Preprocessing ---
                def run_prep():
                    return preprocessor.run_claim_pipe(claim_text)

                claim_title, claim_summary = tracker.run_stage("preprocessor", run_prep)
                claim = Claim(claim_text, claim_title, claim_summary, claim_id=claim_id)

                # --- 2. Retrieval (BYPASSED SCRAPER) ---
                def run_retrieval():
                    mock_srcs = [
                        {
                            "title": "FEVER Perfect Evidence",
                            "url": "https://en.wikipedia.org",
                            "site": "Wikipedia",
                            "body": perfect_evidence,
                        }
                    ]
                    prep_srcs, prep_tokens = preprocessor.run_sources_pipe(mock_srcs)
                    return (prep_srcs, mock_srcs), prep_tokens

                (preprocessed_sources, sources) = tracker.run_stage(
                    "retrieval", run_retrieval
                )
                claim.add_sources(preprocessed_sources)

                # =========================================================
                # 🛡️ SAFETY CHECK: Did Groq find any entities?
                # =========================================================
                has_entities = False
                for src in preprocessed_sources:
                    # Some strings might be literally "[]" or an actual empty list
                    if (
                        src.get("entities")
                        and src.get("entities") != "[]"
                        and len(src.get("entities")) > 0
                    ):
                        has_entities = True
                        break

                if not has_entities:
                    print(
                        "⏩ NER extracted 0 entities. Short-circuiting to prevent Neo4j/Ollama crash."
                    )
                    predicted_label = "Error: No Entities"
                    query_result = "VERDICT: NOT ENOUGH INFO\nREASONING: Evidence was provided, but the NER model failed to extract any entities to build a graph."

                    Answer(claim_id=claim.id, answer=query_result, graphs_folder=None)
                    tracker.finalize(
                        predicted_label,
                        {
                            "claim_text": claim_text,
                            "raw_sources": sources,
                            "query_result": query_result,
                        },
                    )
                    successful_runs += 1
                    continue  # Jump to the next claim!
                # =========================================================

                # --- 3. GraphRAG ---
                def run_rag():
                    q_res, g_folder, t_usage = rag.run_pipeline(
                        preprocessed_sources, claim.text, claim.id
                    )
                    return (q_res, g_folder), t_usage

                (query_result, graphs_folder) = tracker.run_stage("graph_rag", run_rag)

                # --- 4. Verdict Parsing ---
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

                print(f"FoxAI Verdict: {predicted_label}")

                # --- Create Answer Entity for the UI ---
                Answer(
                    claim_id=claim.id, answer=query_result, graphs_folder=graphs_folder
                )

                # --- 5. Log Experiment Metrics ---
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "raw_sources": sources,
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                print("Sleeping for 15 seconds before next claim...")
                time.sleep(
                    15
                )  # Sleep to simulate time taken and avoid overwhelming resources

    except FileNotFoundError:
        print(f"ERROR: Could not find dataset at {DATASET_PATH}.")
    finally:
        wiki_conn.close()

    print("\n" + "=" * 40)
    print("CONTROLLED EXPERIMENT COMPLETE!")
    print(f"Successfully processed: {successful_runs}")
    print("=" * 40)


if __name__ == "__main__":
    run_controlled_experiment()
