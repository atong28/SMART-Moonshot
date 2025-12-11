"""
Utility to add file handlers to existing loggers.
"""
import logging
import os


def setup_file_logging(logger: logging.Logger, file_path: str) -> None:
    """
    Add a file handler to an existing logger.
    
    Args:
        logger: The logger to add file logging to
        file_path: Path to the log file (directory will be created if needed)
    
    Example:
        from ..log import get_logger, setup_file_logging
        
        logger = get_logger(__file__)
        setup_file_logging(logger, "/path/to/logs.txt")
    """
    # Check if file handler already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(file_path):
            return  # Already has this file handler
    
    # Create directory if needed
    log_dir = os.path.dirname(file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

