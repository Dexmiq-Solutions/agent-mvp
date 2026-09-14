"""Context Assembly stage organizing hydrated chunk records into structured retrieval context.

Positions directly between Chunk Fetching / Hydration and downstream Relevance Check / Fallback.
"""

from exceptions.retrieval import (
    ContextAssemblyError,
    ContextAssemblyValidationError,
)
from retrieval.assembly.config import ContextAssemblyConfig
from retrieval.assembly.service import (
    ContextAssemblyService,
    assemble_context,
    assemble_context_async,
    get_context_assembly_service,
    reset_context_assembly_service,
)
from retrieval.models import (
    AssembledContext,
    AssembledContextItem,
    ContextItem,
    RetrievalContext,
    StructuredRetrievalContext,
)

__all__ = [
    # Configuration
    "ContextAssemblyConfig",
    # Service & Functional Helpers
    "ContextAssemblyService",
    "get_context_assembly_service",
    "reset_context_assembly_service",
    "assemble_context",
    "assemble_context_async",
    # Domain Models
    "AssembledContextItem",
    "ContextItem",
    "AssembledContext",
    "RetrievalContext",
    "StructuredRetrievalContext",
    # Exceptions
    "ContextAssemblyError",
    "ContextAssemblyValidationError",
]
