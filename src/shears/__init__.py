import logging
from importlib.metadata import version

from . import deseq, get, pl, pp, tl, util
from ._settings import settings

# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["deseq", "get", "pl", "pp", "tl", "util"]

__version__ = version("shears")
