import logging
import os
from typing import Optional


def setup_logging(logger_name: Optional[str] = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with a consistent format.

    This avoids multiple basicConfig calls across modules.
    """
    logger = logging.getLogger(logger_name if logger_name else "")
    if not logger.handlers:
        handler = logging.StreamHandler()
        log_format = os.getenv(
            "DC_DEL_LOG_FORMAT",
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
