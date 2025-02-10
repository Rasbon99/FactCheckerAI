from selenium import webdriver
from unittest import TestCase
from selenium.webdriver.chrome.options import Options
import time
from log import Logger  # Assumendo che tu abbia una classe Logger
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

class SeleniumTests:
    def __init__(self):
        # Inizializza il logger
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.logger.info("SeleniumTests object created")

        # Configura Selenium in modalità headless
        self.options = Options()
        self.options.add_argument("--headless")

    def test_UI_is_working(self):
        """Verifies that the UI is working correctly by checking the presence of the "FOX AI" text in the page source."""
        self.logger.info("Starting UI test: test_UI_is_working")
        
        # Avvia il browser in modalità headless
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")

        # Attendi il caricamento della pagina
        time.sleep(5)

        # Controlla che il testo "FOX AI" sia presente nella pagina
        assert ("Chat History" in driver.page_source or "Search claims..." in driver.page_source), "Error: FOX AI not found in page source"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_UI_is_working completed successfully")
    
    def test_insert_claim(self):
        """Tests the insertion of a claim in the UI."""
        self.logger.info("Starting UI test: test_insert_claim")
        
        # Avvia il browser in modalità headless
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")

        # Attendi il caricamento della pagina
        time.sleep(5)

        # Inserisci una claim
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("Halloween is a cult horror movie and is the best horror movie ever made")
        claim_input.send_keys(Keys.ENTER)

        # Attendi il caricamento della pagina
        time.sleep(5)

        # Controlla che la claim inserita sia presente nella pagina
        assert "Processing claim..." in driver.page_source, "Error: claim is not being processed"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_insert_claim completed successfully")
    

    def test_answer_confirm(self):
        """Test if the answer is correctly created and confirms the claim."""
        # SETUP START: INSERT A CLAIM AND WAIT FOR THE ANSWER TO BE CREATED
        self.logger.info("Starting UI test: test_answer_confirm")
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        
        time.sleep(5)
        
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("Trees use their roots to drink water from the terrain")
        claim_input.send_keys(Keys.ENTER)
        
        time.sleep(100)
        # SETUP END
        
        assert ("confirmed" in driver.page_source or "Sources" in driver.page_source or "Trees use their roots to drink water from the terrain" in driver.page_source), "Error: claim is not confirmed or the answer is not created"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_answer_confirm completed successfully")
        
    def test_answer_neutral(self):
        """Test if the answer is correctly created and remains neutral."""
        # SETUP START: INSERT A CLAIM AND WAIT FOR THE ANSWER TO BE CREATED
        self.logger.info("Starting UI test: test_answer_neutral")
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        
        time.sleep(5)
        
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("Halloween is the best cult horror movie ever made")
        claim_input.send_keys(Keys.ENTER)
        
        time.sleep(100)
        # SETUP END
        
        assert ("Halloween is the best cult horror movie ever made" in driver.page_source or "not confirmed or refuted" in driver.page_source or "I don't know" in driver.page_source or "Sources" in driver.page_source), "Error: claim is not neutral or the answer is not created"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_answer_neutral completed successfully")
        
    def test_answer_deny(self):
        """Test if the answer is correctly created and denies the claim."""
        # SETUP START: INSERT A CLAIM AND WAIT FOR THE ANSWER TO BE CREATED
        self.logger.info("Starting UI test: test_answer_deny")
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        
        time.sleep(5)
        
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("The sun hovers around the earth")
        claim_input.send_keys(Keys.ENTER)
        
        time.sleep(100)
        # SETUP END
        
        assert ("The sun hovers around the earth" in driver.page_source or "refuted" in driver.page_source or "denied" in driver.page_source or "neither confirmed nor refuted" in driver.page_source or "Sources" in driver.page_source), "Error: claim is not denied or the answer is not created"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_answer_deny completed successfully")
    
    def test_delete_chat_history(self):
        """Test if the chat history is correctly deleted."""
        # SETUP START: INSERT A CLAIM AND WAIT FOR THE ANSWER TO BE CREATED
        self.logger.info("Starting UI test: test_delete_chat_history")
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        
        time.sleep(5)
        
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("The sun hovers around the earth")
        claim_input.send_keys(Keys.ENTER)
        bottone = driver.find_element(By.XPATH, "//div[@data-testid='stMarkdownContainer']/p[contains(text(), '🗑️')]")
        time.sleep(100)
        # SETUP END
        
        #Trova il bottone per testo visibile
        
        bottone.click()
        
        time.sleep(5)
        
        assert ("The sun hovers around the earth" not in driver.page_source and "Chat history deleted successfully."), "Error: chat history is not deleted"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_delete_chat_history completed successfully")
    
    def test_exit_dashboard(self):
        """ Test if the exit button redirects to the dashboard."""
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        time.sleep(5)
        
        button = driver.find_element(By.XPATH, "//div[@data-testid='stMarkdownContainer']/p[contains(text(), '❌')]")
        button.click()
        
        time.sleep(3)
        
        assert "Chat History" not in driver.page_source, "Error: exit button does not redirect to the dashboard"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_exit_dashboard completed successfully")
    
    def test_watch_conversation_from_chat_history(self):
        """Test if the conversation is correctly displayed when the user clicks on a chat history item."""
        # SETUP START: INSERT A CLAIM AND WAIT FOR THE ANSWER TO BE CREATED
        self.logger.info("Starting UI test: test_watch_conversation_from_chat_history")
        
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")
        
        
        time.sleep(5)
        
        claim_input = driver.find_element(By.XPATH, '//textarea[@placeholder="Your message"]')
        claim_input.send_keys("Trees use their roots to drink water from the terrain")
        claim_input.send_keys(Keys.ENTER)
        time.sleep(100)
        # SETUP END
        
        #Trova il bottone per testo visibile
        bottone = driver.find_element(By.XPATH, "//div[@data-testid='stMarkdownContainer']/p[contains(text(), 'roots')]")
        bottone.click()
        
        time.sleep(5)
        
        assert ("Trees use their roots to drink water from the terrain" in driver.page_source and "Sources" in driver.page_source), "Error: conversation is not correctly displayed"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_watch_conversation_from_chat_history completed successfully")
        
    def run(self):
        self.test_UI_is_working()
        self.test_insert_claim()
        self.test_answer_confirm()
        self.test_answer_neutral()
        self.test_answer_deny()
        self.test_delete_chat_history()
        self.test_exit_dashboard()
        self.test_watch_conversation_from_chat_history()
        self.logger.info("SeleniumTests run method executed")

if __name__ == "__main__":
    Tests = SeleniumTests()
    Tests.run()
