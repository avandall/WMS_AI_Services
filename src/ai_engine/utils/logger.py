"""
Logging utilities for WMS AI Engine
"""
import logging
from typing import Optional
from datetime import datetime


class WMSLogger:
    """Custom logger for WMS AI Engine"""
    
    def __init__(self, name: str = "wms_ai_engine", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
    
    def info(self, message: str, extra: Optional[dict] = None):
        """Log info message"""
        if extra:
            message = f"{message} | Extra: {extra}"
        self.logger.info(message)
    
    def error(self, message: str, extra: Optional[dict] = None):
        """Log error message"""
        if extra:
            message = f"{message} | Extra: {extra}"
        self.logger.error(message)
    
    def warning(self, message: str, extra: Optional[dict] = None):
        """Log warning message"""
        if extra:
            message = f"{message} | Extra: {extra}"
        self.logger.warning(message)
    
    def debug(self, message: str, extra: Optional[dict] = None):
        """Log debug message"""
        if extra:
            message = f"{message} | Extra: {extra}"
        self.logger.debug(message)


# Global logger instance
logger = WMSLogger()
