import logging
import os
from logging.handlers import TimedRotatingFileHandler
from config import LOG_DIR, LOG_FILE

# Create log directory if it does not exist
os.makedirs(LOG_DIR, exist_ok=True)

# Build full log file path
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE)


def get_logger(name: str = __name__, level=logging.INFO):
    """
    Creates and returns a configured logger.

    Log format:
    [date] [level] [file name] [function name] [line number] [message]

    Rotation Policy:
    - Rotate logs daily at midnight

    Retention Policy:
    - Keep logs for last 30 days
    """

    # Get logger instance (usually module name)
    logger = logging.getLogger(name)

    # Set minimum log level
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:

        # ---------------- FORMATTER ----------------
        formatter = logging.Formatter(
            "[%(asctime)s] "
            "[%(levelname)s] "
            "[%(filename)s] "
            "[%(funcName)s] "
            "[%(lineno)d] "
            "[%(message)s]",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # ---------------- FILE HANDLER ----------------
        # Rotates log file daily and keeps logs for 30 days
        file_handler = TimedRotatingFileHandler(
            LOG_FILE,
            when="midnight",     # rotate at midnight
            interval=1,          # every 1 day
            backupCount=30,      # keep last 30 log files (retention)
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        # ---------------- CONSOLE HANDLER ----------------
        # Prints logs to terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Attach handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Prevent logs from propagating to root logger
        logger.propagate = False

    return logger
