import unittest
from unittest.mock import patch, MagicMock
from GraphRAG.graph_manager import GraphManager
from GraphRAG.query_engine import QueryEngine
import requests

class TestGraphManager(unittest.TestCase):

    @patch("graph_manager.Neo4jGraph")
    @patch("graph_manager.requests.get")
    def setUp(self, mock_requests, mock_neo4j):
        """Set up mock objects for testing."""
        self.mock_graph = mock_neo4j.return_value
        mock_requests.return_value.status_code = 200
        self.graph_manager = GraphManager()

    @patch("graph_manager.requests.get")
    def test_neo4j_running(self, mock_requests):
        """Test if Neo4j server is running."""
        mock_requests.return_value.status_code = 200
        self.assertTrue(self.graph_manager._is_neo4j_running())


    def test_reset_data(self):
        """Test database reset functionality."""
        self.graph_manager.reset_data()
        self.mock_graph.query.assert_any_call("MATCH (a:Article) DETACH DELETE a")

    def test_load_valid_data(self):
        """Test loading valid data into Neo4j."""
        test_data = [{"title": "Test Article", "url": "http://test.com", "body": "Test Body", "site": "TestSite", "entities": ["Entity1"], "topic": "Topic1"}]
        self.graph_manager.load_data(test_data)
        self.mock_graph.query.assert_called()



    @patch("graph_manager.Graph.run")
    def test_extract_and_save_graph(self, mock_run):
        """Test extraction and saving of graphs."""
        mock_run.return_value.to_data_frame.return_value = [{"Article": "Test", "Topic": "TestTopic"}]
        with patch("matplotlib.pyplot.savefig"):
            self.graph_manager.extract_and_save_graph("topic.jpg", "entity.jpg", "site.jpg")
        self.assertTrue(mock_run.called)
        
        
class TestQueryEngine(unittest.TestCase):

    @patch('query_engine.dotenv.load_dotenv')
    @patch('query_engine.Logger')
    @patch('query_engine.requests.get')
    def setUp(self, mock_requests_get, mock_logger, mock_load_dotenv):
        # Mock environment variables
        patch.dict('os.environ', {
            'NEO4J_URI': 'http://localhost:7474',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password',
            'MODEL_LLM_NEO4J': 'model_name',
            'GROQ_MODEL_NAME': 'groq_model',
            'OLLAMA_SERVER_URL': 'http://localhost:8000'
        }).start()

        # Mock the response of the Ollama server check
        mock_requests_get.return_value.status_code = 200

        self.query_engine = QueryEngine()

    @patch('query_engine.Neo4jVector.from_existing_graph')
    @patch('query_engine.RetrievalQA.from_chain_type')
    def test_query_similarity(self, mock_from_chain_type, mock_from_existing_graph):
        # Mock the retriever and vector_qa
        mock_retriever = MagicMock()
        mock_from_existing_graph.return_value.as_retriever.return_value = mock_retriever
        mock_vector_qa = MagicMock()
        mock_from_chain_type.return_value = mock_vector_qa

        # Mock the result of the query
        mock_vector_qa.invoke.return_value = {"result": "Mocked result"}

        result = self.query_engine.query_similarity("test query")
        self.assertEqual(result, "Mocked result")

    @patch('query_engine.requests.get')
    def test_is_ollama_running(self, mock_requests_get):
        # Test when Ollama server is running
        mock_requests_get.return_value.status_code = 200
        self.assertTrue(self.query_engine._is_ollama_running())

        # Test when Ollama server is not running
        mock_requests_get.side_effect = requests.exceptions.RequestException
        self.assertFalse(self.query_engine._is_ollama_running())

if __name__ == "__main__":
    unittest.main()
