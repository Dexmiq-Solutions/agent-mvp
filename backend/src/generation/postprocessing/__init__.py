"""Post-processing package providing deterministic normalization for generation results."""

from exceptions.generation import (
    PostProcessingError,
    PostProcessingValidationError,
    StructuredOutputError,
)
from generation.postprocessing.base import BasePostProcessor
from generation.postprocessing.config import PostProcessingConfig
from generation.postprocessing.models import PostProcessedResponse, ProcessedResponse
from generation.postprocessing.service import (
    PostProcessingService,
    get_post_processing_service,
    post_process,
    post_process_async,
    reset_post_processing_service,
)

__all__ = [
    # Domain Models
    "ProcessedResponse",
    "PostProcessedResponse",
    # Configuration
    "PostProcessingConfig",
    # Base Interfaces
    "BasePostProcessor",
    # Service & Functional Entrypoints
    "PostProcessingService",
    "get_post_processing_service",
    "reset_post_processing_service",
    "post_process",
    "post_process_async",
    # Domain Exceptions
    "PostProcessingError",
    "PostProcessingValidationError",
    "StructuredOutputError",
]
