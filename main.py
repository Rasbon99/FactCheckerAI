from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim
from Database.data_entities import Answer
from GraphRAG.rag_pipeline import RAG_Pipeline
from Validator.validator import Validator
from Dashboard.dashboard import DashboardPipeline

def main():

    text = """On Saturday evening, as his plane headed from Las Vegas to Miami during a whirlwind, coast-to-coast first trip since returning to office, US President Donald Trump made his way to the back of Air Force One to talk to gathered reporters. On the in-flight television screens, Fox News was back, having replaced CNN - and the president, fresh from a week in which he upended America's government and ripped up its immigration policies, was feeling confident. "We're getting A-pluses on the work done - and also the amount of work done," he said in response to a question from the BBC. "People are saying it was the most successful first week that anybody can remember a president having," he went on. During a 20-minute conversation with journalists, Trump confirmed he had carried out a late-night purge of several independent watchdogs in government agencies. There was more: the president said he thought the US would "get Greenland" as its own territory; he called on Egypt and Jordan to take in more Palestinians; and he said he had a "very good relationship" with UK Prime Minister Sir Keir Starmer - even though "he's liberal". It was the kind of impromptu question and answer session that Joe Biden rarely did while in office, and the latest sign that everything has changed in Washington and in US politics in the six days since Trump returned to the presidency."""

    claim = Claim(text)
    
    preprocessor = Preprocessing_Pipeline()

    preprocessed_claim = preprocessor.run_claim_pipe(claim.text)

    scraper = Scraper()
    sources = scraper.search_and_extract(preprocessed_claim)

    preprocessed_sources = preprocessor.run_sources_pipe(sources)

    claim.add_sources(preprocessed_sources)

    rag = RAG_Pipeline()
    
    query_result = rag.run_pipeline(preprocessed_sources, preprocessed_claim)
    
    print(query_result)

    validator = Validator()
    
    classification_result = validator.predict([item["body"] for item in preprocessed_sources], claim.text)
    
    print(classification_result)
    
    # TODO: Bisogna validare o meno la generazione della LLM e nel caso fargli rifare l'esecuzione di tutto il codice finchè non si trovano
    
    
    # TODO: Generare oggetto Aswer con la seguente struttura Answer(claim_id, answer, image) dove claim_id è l'id del claim, answer è la risposta generata dalla LLM e image è il grafo generato dalla RAG e gestito come Image (PIL) object
    
    return

if __name__ == "__main__":
    main()
