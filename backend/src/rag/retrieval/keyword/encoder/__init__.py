"""Sparse representation encoders for lexical and keyword retrieval."""

from typing import Optional

from app.core.config import Settings, get_settings
from exceptions.retrieval import SparseEncodingError
from rag.retrieval.keyword.encoder.base import BaseSparseEncoder
from rag.retrieval.keyword.encoder.technical import TechnicalSparseEncoder

__all__ = [
    "BaseSparseEncoder",
    "TechnicalSparseEncoder",
    "get_sparse_encoder",
    "reset_sparse_encoder",
]

_sparse_encoder_instance: Optional[BaseSparseEncoder] = None
_sparse_encoder_cache: dict[tuple[str, str], BaseSparseEncoder] = {}


def get_sparse_encoder(
    strategy: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseSparseEncoder:
    """Factory creating or returning cached BaseSparseEncoder instance.

    Args:
        strategy: Optional strategy identifier override (e.g. 'technical_hash').
        settings: Optional Settings instance override.

    Returns:
        BaseSparseEncoder: Configured sparse encoder instance.

    Raises:
        SparseEncodingError: If an unknown strategy is requested.
    """
    global _sparse_encoder_instance, _sparse_encoder_cache

    resolved_settings = settings or get_settings()
    resolved_strategy = strategy or getattr(
        resolved_settings, "SPARSE_ENCODER_STRATEGY", "technical_hash"
    )
    resolved_version = getattr(
        resolved_settings, "SPARSE_ENCODER_VERSION", "1.0"
    )

    cache_key = (resolved_strategy, resolved_version)
    if cache_key in _sparse_encoder_cache:
        return _sparse_encoder_cache[cache_key]

    if resolved_strategy in ("technical_hash", "hashed_lexical", "default"):
        encoder = TechnicalSparseEncoder(version=resolved_version)
        _sparse_encoder_cache[cache_key] = encoder
        if strategy is None and settings is None:
            _sparse_encoder_instance = encoder
        return encoder

    raise SparseEncodingError(f"Unknown sparse encoder strategy '{resolved_strategy}'.")


def reset_sparse_encoder() -> None:
    """Reset cached singleton instances (primarily for testing)."""
    global _sparse_encoder_instance, _sparse_encoder_cache
    _sparse_encoder_instance = None
    _sparse_encoder_cache.clear()
