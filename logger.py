# ------------------- Standard Library -------------------
import os
import logging
from pathlib import Path

def get_logger(
    name: str = "face_logger",
    log_dir: str = "face_logs",
    log_file_name: str = "face_recognition.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Creates and returns a configured logger that:
    - Logs messages to both console and file
    - Automatically creates the log directory if missing
    - Prevents duplicate handlers if called multiple times

    Args:
        name (str): Logger name (useful when logging across multiple modules)
        log_dir (str): Directory for log file
        log_file_name (str): Name of the log file
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full log file path
    log_file = log_path / log_file_name

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs if parent loggers exist

    # Only add handlers once
    if not logger.handlers:
        # Log format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File handler (UTF-8 encoded)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
