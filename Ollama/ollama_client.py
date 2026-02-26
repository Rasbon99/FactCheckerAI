import subprocess
import psutil
import socket
import platform

from log import Logger

class OllamaClient:
    def __init__(self):
        """
        Initializes the OllamaClient instance.

        This sets up the logger, detects the platform (e.g., macOS or Windows),
        and initializes the process variable to track the server process.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.process = None
        self.platform = platform.system()
    
    def __del__(self):
        """
        Ensures the server is stopped when the OllamaClient instance is deleted.
        """
        self._stop_server()
    
    def start_server(self):
        """
        Starts the Ollama server as a separate background process.

        This method determines the platform (macOS or Windows) and executes the appropriate
        command to start the Ollama server. On macOS, it uses the "ollama serve" command, while
        on Windows, it leverages PowerShell to run the command with elevated privileges.

        Raises:
            FileNotFoundError: If the 'ollama' command is not found in the system PATH.
            Exception: If an unexpected error occurs while starting the server.

        Returns:
            None
        """
        self.logger.info("Starting Ollama server...")
        try:
            if self.platform == "Darwin":
                try:
                    # Start the "ollama serve" command as a separate process
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
                    self.logger.error(f"An unexpected error occurred while starting Ollama server: {e}")
            elif self.platform == "Windows":
                try:
                    # Start the server using PowerShell
                    powershell_command = ('Start-Process "cmd" -ArgumentList "/c ollama serve" -Verb runAs')

                    self.process = subprocess.Popen(["powershell", "-Command", powershell_command])

                    self.logger.info("Ollama server started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'ollama' command not found. Ensure Ollama is installed and in PATH.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting Ollama server: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while starting Ollama server with your platform. "
                              f"Ensure you are on Windows or macOS: {e}")

    def _stop_server(self):
        """
        Stops the Ollama server if it is running.

        This method checks if the server process is active and terminates it gracefully.
        If the process does not terminate, it forcibly kills it. Additionally, it verifies
        if the server is still using the default port (11434) and logs warnings if necessary.

        Raises:
            Exception: If the server fails to stop or terminate.

        Returns:
            None
        """
        def is_process_running(pid):
            """Checks if a process with the given PID is running."""
            return psutil.pid_exists(pid)
        
        def is_port_in_use(port):
            """Checks if a specific port is in use (useful for verifying if the server is active)."""
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
        elif self.platform != "Windows":
            self.logger.warning("Ollama server is not running.")
    
    def _is_port_in_use(self, port):
        """
        Checks if a given port is currently in use.

        Args:
            port (int): The port number to check.

        Returns:
            bool: True if the port is in use; False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0
    
    def is_running(self):
        """
        Checks if the Ollama server is currently running.

        This method verifies if the server is active by checking if the default port (11434) is in use.

        Returns:
            bool: True if the Ollama server is running; False otherwise.
        """
        return self._is_port_in_use(11434)