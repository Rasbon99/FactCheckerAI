from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim
from Database.data_entities import Answer
from GraphRAG.rag_pipeline import RAG_Pipeline

from Dashboard.dashboard import DashboardPipeline


def main():
    #TEST 1: NEWS VERA IN INGLESE MA MOLTO GENERICA (TROPPO)
    #text = """On Saturday evening, as his plane headed from Las Vegas to Miami during a whirlwind, coast-to-coast first trip since returning to office, US President Donald Trump made his way to the back of Air Force One to talk to gathered reporters. On the in-flight television screens, Fox News was back, having replaced CNN - and the president, fresh from a week in which he upended America's government and ripped up its immigration policies, was feeling confident. "We're getting A-pluses on the work done - and also the amount of work done," he said in response to a question from the BBC. "People are saying it was the most successful first week that anybody can remember a president having," he went on. During a 20-minute conversation with journalists, Trump confirmed he had carried out a late-night purge of several independent watchdogs in government agencies. There was more: the president said he thought the US would "get Greenland" as its own territory; he called on Egypt and Jordan to take in more Palestinians; and he said he had a "very good relationship" with UK Prime Minister Sir Keir Starmer - even though "he's liberal". It was the kind of impromptu question and answer session that Joe Biden rarely did while in office, and the latest sign that everything has changed in Washington and in US politics in the six days since Trump returned to the presidency."""

    # TEST 2: NEWS VERA IN INGLESE
    text = """President-elect Donald Trump has reiterated his desire for the US to acquire Greenland and the Panama Canal, calling both critical to American national security. Asked if he would rule out using military or economic force in order to take over the autonomous Danish territory or the Canal, he responded: "No, I can't assure you on either of those two. "But I can say this, we need them for economic security," he told reporters during a wide-ranging news conference at his Mar-a-Lago estate in Florida. Both Denmark and Panama have rejected any suggestion that they would give up territory."""
    
    # TEST 3: NEWS VERA IN ITALIANO
    #text = """Decine di bambini che lo circondano e chiedono un selfie insieme al quale non si sottrae. Jannik Sinner è stato sommerso dall'affetto dei tifosi sulle piste a Plan de Corones, a pochi chilometri da casa dei genitori a Sesto Pusteria. Le immagini hanno fatto il giro del web: il tennista indossa un giubbotto verde e occhialoni. Il tennista azzurro, rientrato da Melbourne a Monaco dopo la vittoria degli Australian Open, aveva annunciato che avrebbe fatto visita alla famiglia."""
    
    # TEST 4: FAKE NEWS FACILE DA RICONOSCERE
    #text = """È entrata nel vivo in tutto il mondo la più grande campagna vaccinale della storia, quella per debellare il Sars-CoV-2. Grandi assenti, per il momento, i bambini. Il motivo, a quanto pare, non è quello che ci è stato detto finora, ovvero la mancata sperimentazione sui minori di 14 anni, ma c’è una spiegazione più semplice. Lo ha rivelato Tedros Adhanom Ghebreyesus, Direttore Generale dell’Organizzazione Mondiale della Sanità: “È eticamente inaccettabile vaccinare un bambino senza premiarlo con un succoso lecca-lecca. Considerando che al mondo ci sono circa 2,2 miliardi di bambini e tenendo conto anche della doppia somministrazione, stiamo parlando di realizzare quasi 4 miliardi e mezzo di lecca-lecca in poco tempo, uno sforzo produttivo che nessuna azienda al momento è in grado di sostenere”."""
    
    # TEST 5: FAKE NEWS DIFFICILE DA RICONOSCERE
    #text = """President Trump announces he will not impose 25% tariffs on Canada and Mexico, nor 10% tariffs on China."""
    
    preprocessor = Preprocessing_Pipeline()

    claim_title, claim_summary = preprocessor.run_claim_pipe(text)

    claim = Claim(text, claim_title, claim_summary)

    scraper = Scraper()
    sources = scraper.search_and_extract(claim_title, num_results=10)

    preprocessed_sources = preprocessor.run_sources_pipe(sources)

    claim.add_sources(preprocessed_sources)

    rag = RAG_Pipeline()
    
    query_result = rag.run_pipeline(preprocessed_sources, claim.summary)
    
    print(query_result)

    return

if __name__ == "__main__":
    main()
