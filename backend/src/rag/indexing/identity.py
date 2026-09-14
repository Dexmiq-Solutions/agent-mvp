"""Deterministic stable point identity generation for vector storage."""

import uuid

from exceptions.indexing import InvalidIndexingInputError

# Dedicated, stable namespace UUID for deterministic RFC 4122 v5 UUID point generation
INDEXING_NAMESPACE = uuid.UUID("a3b4c5d6-e7f8-4901-a234-56789abcdef0")


def generate_point_id(
    project_id: str,
    document_id: str,
    chunk_id: str,
    document_version_id: str | None = None,
) -> str:
    """Generate a deterministic, RFC 4122 v5 UUID string representing a Qdrant vector point.

    Ensures stable point identity across re-indexing runs and bounded updates:
    - Same project, document, version, and chunk always map to the exact same UUID.
    - Different document versions or chunk IDs map to distinct UUIDs.
    - Complies strictly with Qdrant's point ID format (standard UUID string).

    Args:
        project_id: Mandatory project/tenant ID.
        document_id: Source document ID.
        chunk_id: Unique chunk ID within document.
        document_version_id: Optional document version identifier.

    Returns:
        str: Deterministic UUID string.

    Raises:
        InvalidIndexingInputError: If any mandatory identity component is missing or invalid.
    """
    if not isinstance(project_id, str) or not project_id.strip():
        raise InvalidIndexingInputError("project_id must be a non-empty string to generate point ID.")
    if not isinstance(document_id, str) or not document_id.strip():
        raise InvalidIndexingInputError("document_id must be a non-empty string to generate point ID.")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise InvalidIndexingInputError("chunk_id must be a non-empty string to generate point ID.")

    clean_proj = project_id.strip()
    clean_doc = document_id.strip()
    clean_chunk = chunk_id.strip()
    clean_ver = document_version_id.strip() if (document_version_id and document_version_id.strip()) else ""

    seed = f"project:{clean_proj}:doc:{clean_doc}:ver:{clean_ver}:chunk:{clean_chunk}"
    return str(uuid.uuid5(INDEXING_NAMESPACE, seed))
