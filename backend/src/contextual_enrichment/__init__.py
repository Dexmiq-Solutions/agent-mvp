"""Contextual enrichment package for the document indexing pipeline."""

from contextual_enrichment.base import BaseContextProvider, DocumentContext
from contextual_enrichment.models import (
    ContextualEnrichmentConfig,
    ContextualEnrichmentReport,
    ContextuallyEnrichedChunk,
    ContextuallyEnrichedDocument,
    extract_representation_text,
)
from contextual_enrichment.providers.structured import StructuredContextProvider
from contextual_enrichment.service import (
    DocumentContextualEnrichmentService,
    get_contextual_enrichment_service,
    reset_contextual_enrichment_service,
)

__all__ = [
    # Base Abstractions
    "BaseContextProvider",
    "DocumentContext",
    # Models & Configuration
    "ContextualEnrichmentConfig",
    "ContextuallyEnrichedChunk",
    "ContextualEnrichmentReport",
    "ContextuallyEnrichedDocument",
    "extract_representation_text",
    # Providers
    "StructuredContextProvider",
    # Service & Factories
    "DocumentContextualEnrichmentService",
    "get_contextual_enrichment_service",
    "reset_contextual_enrichment_service",
]
