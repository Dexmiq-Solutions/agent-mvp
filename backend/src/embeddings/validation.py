"""Validation utilities for embedding vectors and batch responses."""

import math
from typing import Any, Optional

from exceptions.embedding import EmbeddingResponseValidationError


def validate_embedding_vector(
    vector: Any,
    expected_dim: Optional[int] = None,
) -> list[float]:
    """Validate a single embedding vector.

    Args:
        vector: Candidate vector object.
        expected_dim: Optional expected vector dimensionality.

    Returns:
        list[float]: Validated list of float values.

    Raises:
        EmbeddingResponseValidationError: If vector is empty, contains non-numeric
            or non-finite values (NaN/Inf), or dimensionality mismatches.
    """
    if not isinstance(vector, (list, tuple)):
        raise EmbeddingResponseValidationError(
            f"Embedding vector must be a list or tuple of numbers, got '{type(vector).__name__}'."
        )
    if len(vector) == 0:
        raise EmbeddingResponseValidationError("Embedding vector cannot be empty.")

    validated: list[float] = []
    for idx, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise EmbeddingResponseValidationError(
                f"Vector component at index {idx} must be a numeric float/int, got '{type(val).__name__}'."
            )
        if not math.isfinite(val):
            raise EmbeddingResponseValidationError(
                f"Vector component at index {idx} is non-finite value ({val})."
            )
        validated.append(float(val))

    actual_dim = len(validated)
    if expected_dim is not None and actual_dim != expected_dim:
        raise EmbeddingResponseValidationError(
            f"Vector dimensionality mismatch: expected {expected_dim}, got {actual_dim}."
        )

    return validated


def validate_embedding_batch(
    vectors: Any,
    expected_count: Optional[int] = None,
    expected_dim: Optional[int] = None,
) -> list[list[float]]:
    """Validate a batch of embedding vectors.

    Args:
        vectors: Candidate batch of vectors.
        expected_count: Optional expected number of vectors in the batch.
        expected_dim: Optional expected dimensionality for all vectors.

    Returns:
        list[list[float]]: Validated list of float vectors preserving order.

    Raises:
        EmbeddingResponseValidationError: If vector count mismatches, batch is empty,
            vectors have mismatched internal dimensions, or any vector fails validation.
    """
    if not isinstance(vectors, (list, tuple)):
        raise EmbeddingResponseValidationError(
            f"Batch embeddings must be a list or tuple of vectors, got '{type(vectors).__name__}'."
        )

    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingResponseValidationError(
            f"Mismatched embedding count: expected {expected_count} vectors, got {len(vectors)}."
        )

    if len(vectors) == 0:
        raise EmbeddingResponseValidationError("Embedding batch cannot be empty.")

    validated_vectors: list[list[float]] = []
    common_dim: Optional[int] = expected_dim

    for idx, vec in enumerate(vectors):
        validated_vec = validate_embedding_vector(vec, expected_dim=common_dim)
        if common_dim is None:
            common_dim = len(validated_vec)
        elif len(validated_vec) != common_dim:
            raise EmbeddingResponseValidationError(
                f"Inconsistent vector dimensionality at index {idx}: expected {common_dim}, got {len(validated_vec)}."
            )
        validated_vectors.append(validated_vec)

    return validated_vectors
