"""attic: a local, always-current semantic index of your files, built on kuhaku."""

from .config import Config, load_config
from .indexer import Indexer, Report
from .search import Hit, Searcher

__all__ = ["Config", "Hit", "Indexer", "Report", "Searcher", "load_config"]
__version__ = "0.1.0"
