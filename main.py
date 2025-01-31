import platform

# General imports
from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim
from GraphRAG.rag_pipeline import RAG_Pipeline

# Check the operating system
os_name = platform.system()

if os_name == "Windows":
    print("Operating System: Windows")
    # Import the Windows-specific validator
    from Validator.Windows_Validator import Validator
elif os_name == "Darwin": 
    print("Operating System: macOS (ARM Apple Silicon not tested)") 
    # Import the Intel-specific validator
    from Validator.MacOS_Validator import Validator
else:
    print(f"Operating System {os_name} not supported")

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

    # TODO: Implement the Validator class
    validator = Validator()
    
    validator.run(preprocessed_sources, query_result)
    
    return

if __name__ == "__main__":
    main()
