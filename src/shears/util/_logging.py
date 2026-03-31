import logging
from contextlib import contextmanager

@contextmanager
def _mute_logger(logger_name: str, mute: bool = True):
    """Temporarily elevate a logger's level to suppress internal spam."""
    logger = logging.getLogger(logger_name)
    previous_level = logger.level

    if mute:
        logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        logger.setLevel(previous_level)
