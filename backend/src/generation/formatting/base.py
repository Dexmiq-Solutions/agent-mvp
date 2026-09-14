"""Abstract base class defining the strategy interface for Context Formatters."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from generation.formatting.config import ContextFormattingConfig
from generation.formatting.models import FormattedContext


class BaseContextFormatter(ABC):
    """Abstract base class for all context formatting strategies.

    Follows the Strategy Pattern to allow plugging in diverse representation
    approaches (e.g. structured text, XML, JSON, citation-annotated) without
    altering the retrieval pipeline or Prompt Construction contracts.
    """

    def __init__(self, config: Optional[ContextFormattingConfig] = None) -> None:
        """Initialize formatter with configuration.

        Args:
            config: Optional ContextFormattingConfig instance.
        """
        self._config = config or ContextFormattingConfig()

    @property
    def config(self) -> ContextFormattingConfig:
        """Return active formatter configuration."""
        return self._config

    @abstractmethod
    def format(
        self,
        items: Sequence[Any],
        project_id: Optional[str] = None,
    ) -> FormattedContext:
        """Format an ordered sequence of authoritative context items into a FormattedContext.

        Args:
            items: Ordered sequence of context items (e.g. AssembledContextItem).
            project_id: Optional tenant isolation identifier.

        Returns:
            FormattedContext: Model-readable formatted context representation.
        """
        pass
