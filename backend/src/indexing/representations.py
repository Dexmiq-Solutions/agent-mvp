"""Representation extraction and generation helpers for the RAG indexing pipeline."""

from collections.abc import Sequence
from typing import Any, Optional

from app.core.config import Settings, get_settings
from contextual_enrichment.models import extract_representation_text
from exceptions.indexing import InvalidIndexingInputError
from exceptions.retrieval import SparseEncodingError
from retrieval.keyword.encoder import get_sparse_encoder
from retrieval.keyword.encoder.base import BaseSparseEncoder
from storage.vector.models import SparseVector


def extract_representation_texts(document_or_chunks: Any) -> list[str]:
    """Extract representation-ready text contents from chunks after Contextual Enrichment.

    Consumes the representation-ready content (via to_representation_text, to_embedding_text,
    or contextual_content) established after Contextual Enrichment once in O(n) time,
    ensuring both dense embeddings and sparse representation generators receive identical
    logical representation input.

    Args:
        document_or_chunks: Document object with .chunks attribute or sequence of chunk objects.

    Returns:
        list[str]: Sequence of representation-ready text strings corresponding 1-to-1 with chunks.

    Raises:
        InvalidIndexingInputError: If input is empty or chunks cannot be extracted.
    """
    if document_or_chunks is None:
        raise InvalidIndexingInputError("Input document or chunks cannot be None.")

    chunks = getattr(document_or_chunks, "chunks", None)
    if chunks is None:
        if isinstance(document_or_chunks, (list, tuple)):
            chunks = document_or_chunks
        else:
            raise InvalidIndexingInputError(
                f"Expected document with .chunks or sequence of chunks, got '{type(document_or_chunks).__name__}'."
            )

    if len(chunks) == 0:
        raise InvalidIndexingInputError("Cannot extract representation texts from empty chunks.")

    return [extract_representation_text(c) for c in chunks]


def generate_sparse_representations(
    document_or_chunks: Any,
    encoder: Optional[BaseSparseEncoder] = None,
    settings: Optional[Settings] = None,
) -> list[SparseVector]:
    """Generate sparse vector representations for chunks in the indexing pipeline.

    Represents the explicit Sparse Representation Generation pipeline stage positioned
    directly after Contextual Enrichment and parallel to Embedding Generation.

    Args:
        document_or_chunks: Document object with .chunks attribute or sequence of chunk objects.
        encoder: Optional BaseSparseEncoder instance override. Defaults to configured encoder.
        settings: Optional Settings instance override.

    Returns:
        list[SparseVector]: Sparse vector representations corresponding 1-to-1 with input chunks.

    Raises:
        InvalidIndexingInputError: If chunks input is invalid or empty.
        SparseEncodingError: If sparse vector encoding fails.
    """
    texts = extract_representation_texts(document_or_chunks)
    active_encoder = encoder or get_sparse_encoder(settings=settings or get_settings())
    return active_encoder.encode_documents(texts)


__all__ = [
    "extract_representation_text",
    "extract_representation_texts",
    "generate_sparse_representations",
]
