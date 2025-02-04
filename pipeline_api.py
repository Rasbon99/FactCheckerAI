from fastapi import FastAPI
from pydantic import BaseModel
from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim
from GraphRAG.rag_pipeline import RAG_Pipeline

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/process")
def process_text(input_text: InputText):
    text = input_text.text
    
    preprocessor = Preprocessing_Pipeline()
    claim_title, claim_summary = preprocessor.run_claim_pipe(text)
    claim = Claim(text, claim_title, claim_summary)
    
    scraper = Scraper()
    sources = scraper.search_and_extract(claim_title, num_results=5)
    preprocessed_sources = preprocessor.run_sources_pipe(sources)
    claim.add_sources(preprocessed_sources)
    
    rag = RAG_Pipeline()
    query_result = rag.run_pipeline(preprocessed_sources, claim.summary)
    
    return {"claim_title": claim_title, "claim_summary": claim_summary, "sources": preprocessed_sources, "query_result": query_result}