"""
Logging utility for the topology optimization package.

This module provides a consistent interface for logging messages
across the package with different severity levels.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union


def setup_logger(
    name: str = "pytopo3d",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger with the specified parameters.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int or str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        Path to the log file. If None, no file logging is performed.
    log_to_console : bool
        Whether to also log to the console.
    log_format : str, optional
        Custom log format. If None, a default format is used.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Default format includes timestamp, level, and message
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)-8s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Add file handler if a log file is specified
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Create a default logger for the package
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Parameters
    ----------
    name : str, optional
        Name of the logger. If None, the default package logger is returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name is None:
        return logger
    return logging.getLogger(name)


def set_log_level(level: Union[int, str]) -> None:
    """
    Set the log level for the default logger.

    Parameters
    ----------
    level : int or str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger.setLevel(level)


# Convenience functions to log at different levels
def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    logger.critical(msg, *args, **kwargs)


def config_from_args(args: Dict[str, Any]) -> None:
    """
    Configure logging based on command line arguments.

    Parameters
    ----------
    args : Dict[str, Any]
        Command line arguments dictionary.
    """
    # Set log level if specified
    if hasattr(args, "log_level"):
        set_log_level(args.log_level)

    # Configure log file if specified
    if hasattr(args, "log_file") and args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s - %(message)s")
        file_handler.setFormatter(formatter)

        # Remove any existing file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        logger.addHandler(file_handler)
