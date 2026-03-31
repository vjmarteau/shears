import logging


class ShearsFormatter(logging.Formatter):
    """Custom formatter to mimic scverse logging style."""
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        elif record.levelno == logging.DEBUG:
            self._style._fmt = 'DEBUG: %(message)s'
        else:
            self._style._fmt = '%(levelname)s: %(message)s'
        return super().format(record)


class ShearsConfig:
    """Config manager for shears."""
    
    def __init__(self):
        self._verbosity = 1
        self._setup_logger()

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: int):
        self._verbosity = level
        self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("shears")
        
        level_map = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG
        }
        target_level = level_map.get(self._verbosity, logging.DEBUG)
        
        logger.setLevel(target_level)
        logger.propagate = False
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(ShearsFormatter())
        
        logger.addHandler(handler)

settings = ShearsConfig()
