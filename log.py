import sys
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    _instances = {}

    def __new__(cls, name, log_file="app.log", max_bytes=5 * 1024 * 1024, backup_count=1):
        """
        Implements the singleton pattern for logger names to ensure no duplicate loggers.

        Args:
            name (str): The name of the logger. This will be used to identify the logger.
            log_file (str, optional): The log file path. Default is "app.log".
            max_bytes (int, optional): Maximum size of the log file before rotation. Default is 5MB.
            backup_count (int, optional): Number of backup log files to keep. Default is 1.

        Returns:
            Logger: The singleton Logger instance for the given name.

        Raises:
            None
        """
        if name not in cls._instances:
            cls._instances[name] = super(Logger, cls).__new__(cls)
            cls._instances[name]._initialize(name, log_file, max_bytes, backup_count)
        return cls._instances[name]

    def _initialize(self, name, log_file, max_bytes, backup_count):
        """
        Initializes the logger instance with a rotating file handler and console handler.

        Args:
            name (str): The name of the logger.
            log_file (str): The log file path.
            max_bytes (int): Maximum size of the log file before rotation.
            backup_count (int): Number of backup log files to keep.

        Returns:
            None

        Raises:
            None
        """
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s [%(name)s] - %(levelname)s - %(message)s')

            # File handler (rotating)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)

            # Console handler with UTF-8 encoding and error handling
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace'))

            # Adding handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Returns the logger instance.

        Args:
            None

        Returns:
            logging.Logger: The logger instance for the given name.

        Raises:
            None
        """
        return self.logger