"""
Independent logging module to avoid circular imports.
"""
from .get_logger import get_logger
from .setup_file_logging import setup_file_logging
from .is_main_process import is_main_process

__all__ = ['get_logger', 'setup_file_logging', 'is_main_process']

