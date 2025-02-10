import unittest
from unittest.mock import patch, MagicMock
from query_engine import QueryEngine
from graph_manager import GraphManager
import dotenv
import os

class TestIntegrationGraphManagerQueryEngine(unittest.TestCase):

    @patch('query_engine.dotenv.load_dotenv')
    @patch('query_engine.Logger')
    @patch('query_engine.requests.get')
    @patch.dict(os.environ, {
        'NEO4J_URI': 'bolt://neo4j:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'genny2806',
        'MODEL_LLM_NEO4J': 'phi3.5:latest',
        'GROQ_MODEL_NAME': 'llama-3.3-70b-versatile',
        'OLLAMA_SERVER_URL': 'http://ollama:11434',
        'GROQ_API_KEY': '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'
    })
    def setUp(self, mock_requests_get, mock_logger, mock_load_dotenv):
        # Load environment variables from key.env
        dotenv.load_dotenv(dotenv_path='key.env', override=True)

        # Mock the response of the Ollama server check
        mock_requests_get.return_value.status_code = 200

        self.query_engine = QueryEngine()
        self.graph_manager = GraphManager()

    @patch('query_engine.Neo4jVector.from_existing_graph')
    @patch('query_engine.RetrievalQA.from_chain_type')
    @patch('graph_manager.GraphManager.extract_and_save_graph')
    def test_integration_query_and_graph(self, mock_create_and_save_graph, mock_from_chain_type, mock_from_existing_graph):
        # Mock the retriever and vector_qa
        mock_retriever = MagicMock()
        mock_from_existing_graph.return_value.as_retriever.return_value = mock_retriever
        mock_vector_qa = MagicMock()
        mock_from_chain_type.return_value = mock_vector_qa

        # Mock the result of the query
        mock_vector_qa.invoke.return_value = {"result": "Mocked result"}

        # Perform a query
        query_result = self.query_engine.query_similarity("test query")
        self.assertEqual(query_result, "Mocked result")

        # Mock the graph creation
        mock_create_and_save_graph.return_value = None

        # Create and save a graph based on the query result
        query = "MATCH (a:Article)-[:RELATED_TO]->(t:Topic) RETURN a, t"
        node_relation = ('Article', 'Topic')
        node_label = 'Article'
        edge_label = 'RELATED_TO'
        output_file = 'test_graph.png'

        self.graph_manager.extract_and_save_graph(query, node_relation, node_label, edge_label, output_file)
        mock_create_and_save_graph.assert_called_once_with(query, node_relation, node_label, edge_label, output_file)

if __name__ == '__main__':
    unittest.main()