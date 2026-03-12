import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer, Experiment
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline

backend_app = FastAPI()
db = Database()


# Updated model to include optional ground_truth for future use
class InputText(BaseModel):
    text: str
    ground_truth: Optional[str] = "Not Provided"


@backend_app.post("/run_pipeline")
def process_text(input_text: InputText):
    text = input_text.text

    # --- 1. Preprocessing ---
    start_time = time.time()
    preprocessor = Preprocessing_Pipeline()
    claim_title, claim_summary = preprocessor.run_claim_pipe(text)
    claim = Claim(text, claim_title, claim_summary)
    latency_preprocessor = time.time() - start_time

    # --- 2. Retrieval (Scraper) ---
    start_time = time.time()
    scraper = Scraper()
    sources = scraper.search_and_extract(claim_title, num_results=10)
    preprocessed_sources = preprocessor.run_sources_pipe(sources)
    claim.add_sources(preprocessed_sources)
    latency_retrieval = time.time() - start_time

    # --- 3. GraphRAG ---
    start_time = time.time()
    rag = RAG_Pipeline()
    query_result, graphs_folder, token_usage = rag.run_pipeline(
        preprocessed_sources, claim.text, claim.id
    )
    latency_graph_rag = time.time() - start_time

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
    except Exception as e:
        predicted_label = "Parsing Error"

    # --- 5. Answer Entity (For UI) ---
    try:
        if query_result:
            if "REASONING:" in query_result:
                reasoning = query_result.split("REASONING:")[1].strip()
            else:
                reasoning = query_result.replace("VERDICT:", "").strip()
    except Exception:
        reasoning = query_result

    answer = Answer(claim.id, reasoning, graphs_folder)

    # --- 6. Log Experiment Metrics & Evidence ---
    Experiment(
        claim_id=claim.id,
        predicted_label=predicted_label,  # Supported/Refuted/Not Enough Information or error message
        ground_truth=input_text.ground_truth,  # Defaults to "Not Provided"
        latencies={
            "preprocessor": latency_preprocessor,
            "retrieval": latency_retrieval,
            "graph_rag": latency_graph_rag,
        },
        tokens={
            "total": token_usage["total"],
            "calls": token_usage["calls"],
        },
        evidence_data={
            "claim_text": text,
            "raw_sources": sources,
            "query_result": reasoning,
        },
    )

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
