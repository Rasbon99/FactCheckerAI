import time
from fastapi import FastAPI
from pydantic import BaseModel

from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer, Experiment
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline

backend_app = FastAPI()
db = Database()


class InputText(BaseModel):
    text: str


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
        preprocessed_sources, claim.summary, claim.id
    )
    latency_graph_rag = time.time() - start_time

    answer = Answer(claim.id, query_result, graphs_folder)

    # --- 4. Log Experiment Metrics & Evidence ---
    Experiment(
        claim_id=claim.id,
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
            "query_result": query_result,
        },
    )

    return {
        "claim_title": claim_title,
        "claim_summary": claim_summary,
        "sources": preprocessed_sources,
        "query_result": query_result,
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
