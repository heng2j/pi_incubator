"""
Logging utilities for the pi_incubator package.
"""
import logging
import os
import sys
from typing import Optional

def setup_logger(
    name: str = "pi_incubator",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path to save logs
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger
default_logger = setup_logger("pi_incubator")

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Optional suffix for the logger name
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"pi_incubator.{name}")
    return default_logger 