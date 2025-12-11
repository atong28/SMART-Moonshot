"""
Centralized logging utility that creates per-file loggers and ensures logging only on rank 0.
"""
import inspect
import logging
import os
import sys
from typing import Optional

from .is_main_process import is_main_process


class NullLogger(logging.Logger):
    """A no-op logger that discards all messages for non-main processes."""

    def __init__(self, name: str):
        super().__init__(name, logging.NOTSET)
        # Override all logging methods to do nothing
        self.debug = lambda *args, **kwargs: None
        self.info = lambda *args, **kwargs: None
        self.warning = lambda *args, **kwargs: None
        self.error = lambda *args, **kwargs: None
        self.critical = lambda *args, **kwargs: None
        self.exception = lambda *args, **kwargs: None
        self.log = lambda *args, **kwargs: None


# Cache of configured loggers to avoid duplicate handlers
_logger_cache: dict[str, logging.Logger] = {}


def _file_path_to_logger_name(file_path: str) -> str:
    """Convert file path to logger name.
    
    Examples:
        src/modules/marina/model.py -> modules.marina.model
        /absolute/path/src/modules/data/encoder.py -> modules.data.encoder
    """
    # Normalize path separators
    normalized = file_path.replace(os.sep, '/')
    
    # Find 'src/' or 'modules/' to start from
    if '/src/modules/' in normalized:
        start_idx = normalized.index('/src/modules/') + len('/src/')
    elif '/modules/' in normalized:
        start_idx = normalized.index('/modules/') + 1
    elif normalized.endswith('/modules'):
        # Handle case where file_path is just a directory
        return 'modules'
    else:
        # Fallback: use filename without extension
        return os.path.splitext(os.path.basename(file_path))[0]
    
    # Extract from modules/ onwards
    module_path = normalized[start_idx:]
    
    # Remove .py extension
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    
    # Convert to module-style name (replace / with .)
    logger_name = module_path.replace('/', '.')
    
    return logger_name


def get_logger(file_path: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for the current file, automatically filtering to rank 0 only.
    
    Args:
        file_path: Path to the file (typically __file__). If None, auto-detects from call stack.
    
    Returns:
        logging.Logger: A logger that only logs on rank 0/main process.
        Returns NullLogger (no-op) for non-main processes.
    
    Example:
        from ..log import get_logger
        
        logger = get_logger(__file__)
        logger.info("This only logs on rank 0")
    """
    # Auto-detect __file__ if not provided
    if file_path is None:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            file_path = frame.f_back.f_globals.get('__file__', 'unknown')
        else:
            file_path = 'unknown'
    
    # Convert to logger name
    logger_name = _file_path_to_logger_name(file_path)
    
    # Check cache first
    if logger_name in _logger_cache:
        return _logger_cache[logger_name]
    
    # If not main process, return NullLogger
    if not is_main_process():
        logger = NullLogger(logger_name)
        _logger_cache[logger_name] = logger
        return logger
    
    # Create or get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Only add handlers if not already configured
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Cache and return
    _logger_cache[logger_name] = logger
    return logger

