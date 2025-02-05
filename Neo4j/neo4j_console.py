import subprocess
import psutil
import socket
import platform

from log import Logger

class Neo4jClient:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.process = None
        self.platform = platform.system()

    def __del__(self):
        self._stop_console()

    def _start_console(self):
        """
        Starts the Neo4j console as a background process.
        """
        self.logger.info("Starting Neo4j console...")
        try:
            if self.platform == "Darwin":
                try:
                    # Avvia il comando "neo4j console" in un processo separato
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
        Checks if the given process is still running.
        """
        def is_process_running(pid):
            """Check if a process with a given PID is still running."""
            return psutil.pid_exists(pid)

        def is_port_in_use(port):
            """Check if a specific port is in use (useful for verifying if the console is active)."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        return process and is_process_running(process.pid) or is_port_in_use(7687)  # Assuming Neo4j runs on port 7687


    def _stop_console(self):
        """
        Stops the Neo4j console if it is running.
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
