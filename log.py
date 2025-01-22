import logging
import sys
from logging.handlers import RotatingFileHandler

class Logger:
    _instances = {}

    def __new__(cls, name, log_file="app.log", max_bytes=5 * 1024 * 1024, backup_count=1):
        """
        Implementing singleton pattern per logger name to ensure no duplicate loggers.
        """
        if name not in cls._instances:
            cls._instances[name] = super(Logger, cls).__new__(cls)
            cls._instances[name]._initialize(name, log_file, max_bytes, backup_count)
        return cls._instances[name]

    def _initialize(self, name, log_file, max_bytes, backup_count):
        """
        Initializes the logger instance with a rotating file handler and console handler.
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
        """
        return self.logger
