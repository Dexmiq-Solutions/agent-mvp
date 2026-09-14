"""Abstract base class defining the interface for generation Post-processors."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from generation.postprocessing.config import PostProcessingConfig
from generation.postprocessing.models import ProcessedResponse


class BasePostProcessor(ABC):
    """Abstract base class for all post-processing strategies.

    Follows the Strategy Pattern to decouple deterministic normalization,
    structured output validation, and downstream response preparation
    from concrete LLM providers or future Agent orchestration frameworks.
    """

    def __init__(self, config: Optional[PostProcessingConfig] = None) -> None:
        """Initialize post-processor with configuration.

        Args:
            config: Optional PostProcessingConfig instance.
        """
        self._config = config or PostProcessingConfig()

    @property
    def config(self) -> PostProcessingConfig:
        """Return active post-processor configuration."""
        return self._config

    @abstractmethod
    def process(
        self,
        result: Any,
        **kwargs: Any,
    ) -> ProcessedResponse:
        """Process and normalize raw or application-level generation output.

        Args:
            result: Upstream LLMResult, dict, duck-typed result, or raw string.
            **kwargs: Additional runtime options or overrides.

        Returns:
            ProcessedResponse: Normalized application-level response.
        """
        pass
