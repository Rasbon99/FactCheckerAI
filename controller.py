import os
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from log import Logger
from pydantic import BaseModel

class InputText(BaseModel):
    text: str

# Load environment variables from the key.env file
load_dotenv("key.env")

class Controller:
    def __init__(self):
        # Initialize the logger
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Read the server URLs from environment variables
        self.ollama_server_url = os.getenv("OLLAMA_SERVER_URL", "http://127.0.0.1:8000")
        self.neo4j_server_url = os.getenv("NEO4J_SERVER_URL", "http://127.0.0.1:8002")
        self.backend_server_url = os.getenv("BACKEND_SERVER_URL", "http://127.0.0.1:8001")

        # Create FastAPI instance to expose endpoints
        self.app = FastAPI()

        # Register endpoints using add_api_route
        self.app.add_api_route(
        path="/results",
        endpoint=self.post_results,
        methods=["POST"],
        response_model=dict,
        summary="Call backend run_pipeline API",
        description="Endpoint that accepts a 'text' parameter in the request body and passes it to the backend via the /run_pipeline API."
        )
        self.app.add_api_route(
            path="/clean_conversations",
            endpoint=self.clean_conversations,
            methods=["POST"],
            response_model=dict,
            summary="Clean conversations",
            description="Endpoint that triggers a cleanup of conversations by calling the backend's /delete_db endpoint."
        )
        self.app.add_api_route(
            path="/conversations",
            endpoint=self.get_conversation,
            methods=["GET"],
            response_model=dict,
            summary="Get history of conversations",
            description="Endpoint that returns conversations by calling the backend's /get_history endpoint."
        )

    def post_results(self, input_text: InputText):
        """
        Endpoint POST che accetta un parametro 'text' nel body della richiesta e lo passa al backend tramite l'API /run_pipeline.
        Esempio di chiamata: POST /results con body JSON: {"text": "This is the text to process"}

        Args:
            text (str): Il testo da processare.

        Returns:
            dict: Un dizionario contenente lo status_code e la risposta JSON del backend.
        """
        data = {"text": input_text.text}
        try:
            response = requests.post(f"{self.backend_server_url}/run_pipeline", json=data)
            if response.status_code != 200:
                self.logger.error(f"Error from backend: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return {
                "status_code": response.status_code,
                "response": response.json()
            }
        except Exception as e:
            self.logger.error(f"Error calling run_pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def clean_conversations(self):
        """
        Endpoint that triggers a cleanup of conversations by calling the backend's /delete_db endpoint.
        This endpoint does not return any content.
        """
        try:
            response = requests.post(f"{self.backend_server_url}/delete_db")
            if response.status_code != 200:
                self.logger.error(f"Error from backend on delete_db: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
            # Return an empty response
            return {}
        except Exception as e:
            self.logger.error(f"Error calling delete_db: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_conversation(self):
        """
        Endpoint that returns conversations by calling the backend's /get_history endpoint.
        """
        try:
            response = requests.get(f"{self.backend_server_url}/get_history")
            if response.status_code != 200:
                self.logger.error(f"Error from backend on get_history: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
            # Return an empty response
            response = {
                "status_code": response.status_code,
                "response": response.json()
            }
            return response
        except Exception as e:
            self.logger.error(f"Error calling get_history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _start_servers(self):
        # Start the Ollama server
        try:
            url = f"{self.ollama_server_url}/start"
            requests.post(url)
            self.logger.info("Ollama server started successfully.")
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")

        # Start the Neo4j server
        try:
            url = f"{self.neo4j_server_url}/start"
            requests.post(url)
            self.logger.info("Neo4j server started successfully.")
        except Exception as e:
            self.logger.error(f"Error starting Neo4j server: {e}")

    def stop_servers(self):
        # Method to stop both servers
        try:
            url = f"{self.ollama_server_url}/stop"
            requests.post(url)
            self.logger.info("Ollama server stopped.")
        except Exception as e:
            self.logger.error(f"Error stopping Ollama server: {e}")

        try:
            url = f"{self.neo4j_server_url}/stop"
            requests.post(url)
            self.logger.info("Neo4j server stopped.")
        except Exception as e:
            self.logger.error(f"Error stopping Neo4j server: {e}")

# Create an instance of Controller and retrieve the FastAPI app
controller = Controller()
app = controller.app

# Additional endpoint to stop the servers, if needed:
@app.get("/stop_all")
def stop_all():
    controller.stop_servers()
    return {"message": "All servers have been stopped."}
