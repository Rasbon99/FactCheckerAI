from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from log import Logger  # Assumendo che tu abbia una classe Logger

class SeleniumTests:
    def __init__(self):
        # Inizializza il logger
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.logger.info("SeleniumTests object created")

        # Configura Selenium in modalità headless
        self.options = Options()
        self.options.add_argument("--headless")

    def test_UI_is_working(self):
        """Verifica che l'interfaccia utente sia visibile e contenga 'FOX AI'"""
        self.logger.info("Starting UI test: test_UI_is_working")
        
        # Avvia il browser in modalità headless
        driver = webdriver.Chrome(options=self.options)
        driver.get("http://localhost:8501")

        # Attendi il caricamento della pagina
        time.sleep(5)

        # Controlla che il testo "FOX AI" sia presente nella pagina
        assert "FOX AI" in driver.page_source, "Error: FOX AI not found in page source"
        self.logger.info("Test pass")
        
        driver.quit()
        self.logger.info("test_UI_is_working completed successfully")

    def run(self):
        self.test_UI_is_working()
        self.logger.info("SeleniumTests run method executed")

if __name__ == "__main__":
    Tests = SeleniumTests()
    Tests.run()
