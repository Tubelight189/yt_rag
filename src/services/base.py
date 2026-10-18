"""Base service class."""

from abc import ABC

from src.core.config import Settings, get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseService(ABC):
    """Abstract base class for services."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the service.

        Args:
            settings: Application settings (default from get_settings)
        """
        self.settings = settings or get_settings()
        logger.info(f"Initialized {self.__class__.__name__}")
