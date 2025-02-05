import subprocess
import psutil
import socket

from log import Logger

class OllamaClient:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.process = None
    
    def __del__(self):
        self._stop_server()

    def start_server(self):
        """
        Starts the Ollama server as a separate background process.
        """
        self.logger.info("Starting Ollama server...")
        try:
            # Start the Ollama server in a separate process
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.logger.info("Ollama server started successfully as a background process.")
        except FileNotFoundError:
            self.logger.error("Error: 'ollama' command not found. Ensure Ollama is installed and in PATH.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while starting the server: {e}")

    def _stop_server(self):
        """
        Stops the Ollama server if it is running.
        """
        def is_process_running(pid):
            """Check if a process with a given PID is still running."""
            return psutil.pid_exists(pid)
        
        def is_port_in_use(port):
            """Check if a specific port is in use (useful for verifying if the server is active)."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0
        
        if self.process and is_process_running(self.process.pid):  # Check if the process is still active
            self.logger.info("Stopping the Ollama server...")
            self.process.terminate()  # Send a terminate signal
            self.process.wait()  # Wait for the process to terminate
            
            # Double-check if the process is still running
            if is_process_running(self.process.pid) or is_port_in_use(11434):  # Assuming Ollama runs on port 11434
                self.logger.warning("Failed to stop the Ollama server. Forcing termination...")
                self.process.kill()  # Force kill the process
            else:
                self.logger.info("Ollama server stopped successfully.")
        else:
            self.logger.warning("Ollama server is not running.")
    
    def is_running(self):
        """
        Checks if the Ollama server is currently running.
        """
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0
        
        if self.process and psutil.pid_exists(self.process.pid):
            return True
        elif is_port_in_use(11434):  # Assuming Ollama runs on port 11434
            return True
        return False
