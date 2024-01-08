"""
This script sets up a customized logging system using Python's logging module.
It creates separate handlers for console and file outputs with different logging levels and formats.
Logs are stored in a specified directory and are rotated when they reach a certain size.
"""

import logging
import logging.config
import sys
from pathlib import Path
from rich.logging import RichHandler  # Import RichHandler

# Set the LOGS_DIR variable
LOGS_DIR = './log/'  # Replace with the path to your logs directory
Path(LOGS_DIR).mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# Define your logging configuration dictionary
logging_config = {
    "version": 1,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# Apply the logging configuration using dictConfig
logging.config.dictConfig(logging_config)

# Create a logger instance
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set RichHandler as the handler for the root logger
logger.root.handlers[0] = RichHandler(markup=True)

# Logging levels (from lowest to highest priority)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong, and the process may terminate.")