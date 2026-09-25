import os
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from log import Logger
from pydantic import BaseModel


class InputText(BaseModel):
    """
    Data model for the input text used in POST requests.

    Attributes:
        text (str): The text to be processed.
    """

    text: str


# Load environment variables from the key.env file
load_dotenv("key.env")


class Controller:
    def __init__(self):
        """
        Initializes the Controller instance.

        This sets up the logger, reads environment variables for server URLs, initializes
        the FastAPI app, and registers the API routes.
        """
        # Initialize the logger
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Read the server URLs from environment variables
        self.backend_server_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8001")

        # Create FastAPI instance to expose endpoints
        self.app = FastAPI()

        # Register endpoints using add_api_route
        self.app.add_api_route(
            path="/results",
            endpoint=self.post_results,
            methods=["POST"],
            response_model=dict,
            summary="Call backend run_pipeline API",
            description="Endpoint that accepts a 'text' parameter in the request body and passes it to the backend via the /run_pipeline API.",
        )
        self.app.add_api_route(
            path="/clean_conversations",
            endpoint=self.clean_conversations,
            methods=["POST"],
            response_model=dict,
            summary="Clean conversations",
            description="Endpoint that triggers a cleanup of conversations by calling the backend's /delete_db endpoint.",
        )
        self.app.add_api_route(
            path="/conversations",
            endpoint=self.get_conversation,
            methods=["GET"],
            response_model=dict,
            summary="Get history of conversations",
            description="Endpoint that returns conversations by calling the backend's /get_history endpoint.",
        )

    def post_results(self, input_text: InputText):
        """
        Processes a text by calling the backend's /run_pipeline API.

        Args:
            input_text (InputText): The input text to process, passed as a JSON body.

        Returns:
            dict: A dictionary containing the status code and response JSON from the backend.

        Raises:
            HTTPException: If the backend returns an error or if the request fails.
        """
        data = {"text": input_text.text}
        try:
            response = requests.post(
                f"{self.backend_server_url}/run_pipeline", json=data
            )
            if response.status_code != 200:
                self.logger.error(f"Error from backend: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )
            return {"status_code": response.status_code, "response": response.json()}
        except Exception as e:
            self.logger.error(f"Error calling run_pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def clean_conversations(self):
        """
        Cleans conversations by calling the backend's /delete_db API.

        Returns:
            dict: An empty dictionary to indicate successful cleanup.

        Raises:
            HTTPException: If the backend returns an error or if the request fails.
        """
        try:
            response = requests.post(f"{self.backend_server_url}/delete_db")
            if response.status_code != 200:
                self.logger.error(f"Error from backend on delete_db: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )
            # Return an empty response
            return {}
        except Exception as e:
            self.logger.error(f"Error calling delete_db: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_conversation(self):
        """
        Retrieves conversation history by calling the backend's /get_history API.

        Returns:
            dict: A dictionary containing the status code and response JSON from the backend.

        Raises:
            HTTPException: If the backend returns an error or if the request fails.
        """
        try:
            response = requests.get(f"{self.backend_server_url}/get_history")
            if response.status_code != 200:
                self.logger.error(f"Error from backend on get_history: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )
            # Return an empty response
            response = {
                "status_code": response.status_code,
                "response": response.json(),
            }
            return response
        except Exception as e:
            self.logger.error(f"Error calling get_history: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Create an instance of Controller and retrieve the FastAPI app
controller = Controller()
app = controller.app
