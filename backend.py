import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline
from Evaluation.Utils.experiment_tracker import ExperimentTracker

backend_app = FastAPI()
db = Database()


# Updated model to include optional ground_truth for future use
class InputText(BaseModel):
    text: str
    ground_truth: Optional[str] = "Not Provided"
    dataset_setting: Optional[str] = "User-Query"


@backend_app.post("/run_pipeline")
def process_text(input_text: InputText):
    text = input_text.text

    # Generate a claim ID upfront so the tracker and the Claim entity match perfectly
    claim_id = str(uuid.uuid4())

    # --- Initialize Components & Tracker ---
    tracker = ExperimentTracker(
        claim_id=claim_id,
        ground_truth=input_text.ground_truth or "Not Provided",
        system_type="FoxAI-GraphRAG",
        dataset_setting=input_text.dataset_setting or "User-Query",
    )

    preprocessor = Preprocessing_Pipeline()
    scraper = Scraper()
    rag = RAG_Pipeline()

    # --- 1. Preprocessing ---
    def run_prep():
        return preprocessor.run_claim_pipe(text)

    claim_title, claim_summary = tracker.run_stage("preprocessor", run_prep)

    claim = Claim(text, claim_title, claim_summary, claim_id=claim_id)

    # --- 2. Retrieval (Scraper + Preprocessor) ---
    def run_retrieval():
        srcs, scraper_metrics = scraper.search_and_extract(claim_title, num_results=10)

        prep_srcs, prep_metrics = preprocessor.run_sources_pipe(srcs)

        # Combine the dictionaries mathematically
        total_retrieval_tokens = scraper_metrics["total"] + prep_metrics["total"]
        total_retrieval_calls = scraper_metrics["calls"] + prep_metrics["calls"]

        # Pass the unified dictionary to the tracker!
        return (prep_srcs, srcs), {
            "total": total_retrieval_tokens,
            "calls": total_retrieval_calls,
        }

    preprocessed_sources, sources = tracker.run_stage("retrieval", run_retrieval)

    # --- 3. GraphRAG ---
    def run_rag():
        q_res, g_folder, t_usage = rag.run_pipeline(
            preprocessed_sources, claim.text, claim.id
        )
        # We group the first two outputs so the tracker sees exactly (result, tokens)
        return (q_res, g_folder), t_usage

    query_result, graphs_folder = tracker.run_stage("generation", run_rag)

    # --- 4. Verdict Parsing for RQ1 ---
    # Extract the "Supported/Refuted/NEI" label from the structured response
    try:
        if query_result and "VERDICT:" in query_result:
            # Takes the text between VERDICT: and REASONING:
            predicted_label = (
                query_result.split("REASONING:")[0].replace("VERDICT:", "").strip()
            )
        else:
            # Fallback if the LLM didn't follow the format or if query_result is None
            predicted_label = "Error: Unstructured Response"
    except Exception:
        predicted_label = "Parsing Error"

    # --- 5. Answer Entity (For UI) ---
    try:
        if query_result:
            if "REASONING:" in query_result:
                reasoning = query_result.split("REASONING:")[1].strip()
            else:
                reasoning = query_result.replace("VERDICT:", "").strip()
        else:
            reasoning = "No results found."
    except Exception:
        reasoning = query_result

    answer = Answer(claim.id, reasoning, graphs_folder)

    # --- 6. Log Experiment Metrics & Evidence ---
    evidence_data = {
        "claim_text": text,
        "raw_sources": sources,
        "query_result": reasoning,
    }

    tracker.finalize(predicted_label, evidence_data)

    return {
        "claim_title": claim_title,
        "claim_summary": claim_summary,
        "sources": preprocessed_sources,
        "query_result": reasoning,
        "graphs_folder": graphs_folder,
        "answer": answer.answer,
    }


@backend_app.post("/delete_db")
def delete_database():
    db.delete_all_conversations()


@backend_app.get("/get_history")
def get_database():
    history = db.get_history()
    return history
