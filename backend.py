import uuid
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline

backend_app = FastAPI()
db = Database()


class InputText(BaseModel):
    text: str
    search_query: Optional[str] = None
    prompt_instructions: Optional[str] = None


@backend_app.post("/run_pipeline")
def process_text(input_text: InputText):
    text = input_text.text
    claim_id = str(uuid.uuid4())

    preprocessor = Preprocessing_Pipeline()
    scraper = Scraper()
    rag = RAG_Pipeline()

    latencies = {}
    tokens = {}
    calls = {}

    # --- 1. Preprocessing ---
    t0 = time.time()

    prep_data, prep_claim_metrics = preprocessor.run_claim_pipe(text)
    claim_title, claim_summary = prep_data

    # SAFETY FALLBACK: If LLM fails to summarize, use the raw claim text
    if not claim_title:
        claim_title = f"!g {text[:50]}..."
    if not claim_summary:
        claim_summary = text

    latencies["preprocessor"] = time.time() - t0

    tokens["preprocessor"] = prep_claim_metrics.get("total", 0)
    calls["preprocessor"] = prep_claim_metrics.get("calls", 0)

    claim = Claim(text, claim_title, claim_summary, claim_id=claim_id)

    target_search_query = (
        input_text.search_query if input_text.search_query else claim_title
    )

    # --- 2. Retrieval (Scraper + Preprocessor) ---
    t0 = time.time()
    sources, scraper_metrics = scraper.search_and_extract(
        target_search_query, num_results=10
    )
    preprocessed_sources, prep_metrics = preprocessor.run_sources_pipe(sources)

    claim.add_sources(preprocessed_sources)

    latencies["retrieval"] = time.time() - t0

    tokens["retrieval"] = scraper_metrics.get("total", 0) + prep_metrics.get("total", 0)
    calls["retrieval"] = scraper_metrics.get("calls", 0) + prep_metrics.get("calls", 0)

    # --- 3. GraphRAG ---
    t0 = time.time()
    query_result, graphs_folder, t_usage = rag.run_pipeline(
        preprocessed_sources,
        claim.text,
        claim.id,
        prompt_instructions=input_text.prompt_instructions,
    )
    latencies["generation"] = time.time() - t0

    tokens["generation"] = t_usage.get("llm_total", t_usage.get("total", 0))
    calls["generation"] = t_usage.get("llm_calls", t_usage.get("calls", 0))

    # --- 4. Verdict Parsing ---
    try:
        if query_result and "VERDICT:" in query_result:
            predicted_label = (
                query_result.split("REASONING:")[0].replace("VERDICT:", "").strip()
            )
        else:
            predicted_label = "Error: Unstructured Response"
    except Exception:
        predicted_label = "Parsing Error"

    # --- 5. Answer Entity ---
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

    evidence_data = {
        "claim_text": text,
        "raw_sources": sources,
        "query_result": reasoning,
    }

    return {
        "claim_id": claim_id,
        "claim_title": claim_title,
        "claim_summary": claim_summary,
        "sources": preprocessed_sources,
        "query_result": reasoning,
        "predicted_label": predicted_label,
        "graphs_folder": graphs_folder,
        "answer": answer.answer,
        "metrics": {"latencies": latencies, "tokens": tokens, "calls": calls},
        "evidence_data": evidence_data,
    }


@backend_app.post("/delete_db")
def delete_database():
    db.delete_all_conversations()


@backend_app.get("/get_history")
def get_database():
    history = db.get_history()
    return history
