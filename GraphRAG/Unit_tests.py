import unittest
from unittest.mock import patch, MagicMock
from graph_manager import GraphManager

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
        
        


if __name__ == "__main__":
    unittest.main()
