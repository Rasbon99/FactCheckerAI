import subprocess
import psutil
import socket
import platform

from log import Logger

class Neo4jClient:
    def __init__(self):
        """
        Initializes the Neo4jClient object.

        This includes setting up a logger, initializing the process attribute, and detecting the platform.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.process = None
        self.platform = platform.system()

    def __del__(self):
        """
        Destructor for the Neo4jClient object.

        Ensures the Neo4j console is stopped when the object is deleted.
        """
        self._stop_console()

    def _start_console(self):
        """
        Starts the Neo4j console as a background process.

        This method uses platform-specific commands to start the Neo4j console. On macOS, it uses the `neo4j console` command,
        while on Windows, it starts a PowerShell process to run the command.

        Raises:
            FileNotFoundError: If the `neo4j` command is not found.
            Exception: For any unexpected errors during process execution.
        """
        self.logger.info("Starting Neo4j console...")
        try:
            if self.platform == "Darwin":
                try:
                    # Start the "neo4j console" command in a separate process
                    self.process = subprocess.Popen(
                        ["neo4j", "console"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.logger.info("Neo4j console started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'neo4j' command not found.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting Neo4j console: {e}")
            elif self.platform == "Windows":
                try:
                    powershell_command = ('Start-Process "cmd" -ArgumentList "/c neo4j console" -Verb runAs')

                    self.process = subprocess.Popen(["powershell", "-Command", powershell_command])

                    self.logger.info("Neo4j console started successfully as a background process.")
                except FileNotFoundError:
                    self.logger.error("Error: 'neo4j' command not found.")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred while starting Neo4j console: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while starting Neo4j console with your platform, make sure you are on Windows or macOS: {e}")

    def is_running(self, process):
        """
        Checks if the Neo4j process or port is still active.

        Args:
            process (subprocess.Popen): The process object to check.

        Returns:
            bool: True if the process is running or if the Neo4j port (default: 7687) is in use; False otherwise.
        """
        def is_process_running(pid):
            """
            Checks if a process with the given PID is still running.

            Args:
                pid (int): The process ID.

            Returns:
                bool: True if the process exists; False otherwise.
            """
            return psutil.pid_exists(pid)

        def is_port_in_use(port):
            """
            Checks if a specific port is in use.

            Args:
                port (int): The port number to check.

            Returns:
                bool: True if the port is in use; False otherwise.
            """
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        return process and is_process_running(process.pid) or is_port_in_use(7687)  # Assuming Neo4j runs on port 7687

    def _stop_console(self):
        """
        Stops the Neo4j console if it is currently running.

        This method sends a terminate signal to the process and ensures it is properly stopped. 
        If the process does not terminate, it forces termination.

        Raises:
            Warning: If the Neo4j console is not running or fails to stop.
        """
        if self.is_running(self.process):  # Check if the process is still active
            self.logger.info("Stopping Neo4j console...")
            self.process.terminate()  # Send a terminate signal
            self.process.wait()  # Wait for the process to terminate

            # Double-check if the process is still running
            if self.is_running(self.process):
                self.logger.warning("Failed to stop the Neo4j console. Forcing termination...")
                self.process.kill()  # Force kill the process
            else:
                self.logger.info("Neo4j console stopped successfully.")
        elif self.platform != "Windows":
            self.logger.warning("Neo4j console is not running.")