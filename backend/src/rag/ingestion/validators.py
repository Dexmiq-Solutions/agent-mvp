"""Validation and document type resolution for the document ingestion layer."""

from pathlib import PurePosixPath
from typing import Any

from rag.acquisition.formats import (
    EXTENSION_TO_DOCUMENT_TYPE,
    DocumentType,
    get_canonical_mime_type,
    normalize_extension,
)
from exceptions.ingestion import (
    InvalidIngestionInputError,
    UnsupportedDocumentTypeError,
)

# Supported MIME types mapping directly to DocumentType
MIME_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/msword": DocumentType.DOCX,
    "text/markdown": DocumentType.MARKDOWN,
    "text/x-markdown": DocumentType.MARKDOWN,
    "text/plain": DocumentType.TXT,
}

# Generic MIME types that should defer to file extension inspection
GENERIC_MIME_TYPES: frozenset[str] = frozenset({
    "application/octet-stream",
    "binary/octet-stream",
    "application/download",
    "application/x-download",
    "",
})


def validate_project_id(project_id: Any) -> str:
    """Validate and normalize a project identifier to enforce tenant isolation.
    
    Args:
        project_id: Project identifier to validate.
        
    Returns:
        Clean normalized project identifier string.
        
    Raises:
        InvalidIngestionInputError: If project_id is empty, non-string, or attempts traversal.
    """
    if not isinstance(project_id, str) or not project_id.strip():
        raise InvalidIngestionInputError(
            "project_id must be a non-empty string to enforce project isolation."
        )
    clean = project_id.strip().replace("\\", "/").strip("/")
    if not clean or "/" in clean or ".." in clean:
        raise InvalidIngestionInputError(
            f"Invalid project_id '{project_id}': must be a single path segment without traversal."
        )
    return clean


def normalize_storage_path(path: Any) -> str:
    """Normalize and validate a storage path string.
    
    Args:
        path: Path string to normalize.
        
    Returns:
        Normalized relative POSIX storage path.
        
    Raises:
        InvalidIngestionInputError: If path is empty, non-string, or invalid.
    """
    if not isinstance(path, str) or not path.strip():
        raise InvalidIngestionInputError("Storage path cannot be empty.")
    clean = path.strip().replace("\\", "/").strip("/")
    if not clean:
        raise InvalidIngestionInputError("Storage path cannot resolve to an empty string.")
    if ".." in clean.split("/"):
        raise InvalidIngestionInputError(f"Path traversal not permitted in storage path '{path}'.")
    return clean


def resolve_and_verify_storage_path(project_id: str, storage_path: str) -> str:
    """Verify and resolve full storage path ensuring strict project tenant isolation.
    
    Args:
        project_id: Project identifier.
        storage_path: Relative or project-prefixed storage path.
        
    Returns:
        Full project-scoped storage path.
        
    Raises:
        InvalidIngestionInputError: If path attempts cross-tenant access or points to project root.
    """
    clean_project_id = validate_project_id(project_id)
    clean_path = normalize_storage_path(storage_path)

    project_prefix = f"{clean_project_id}/"
    if clean_path.startswith(project_prefix):
        return clean_path
    if clean_path == clean_project_id:
        raise InvalidIngestionInputError(
            f"Storage path '{clean_path}' points to project root, not a document."
        )

    parts = clean_path.split("/")
    if len(parts) > 1 and parts[0] != clean_project_id:
        raise InvalidIngestionInputError(
            f"Cross-project access prohibited: path '{clean_path}' does not belong to project '{clean_project_id}'."
        )

    return f"{clean_project_id}/{clean_path}"


def resolve_and_validate_document_type(
    content_type: str | None = None,
    filename_or_path: str | None = None,
    explicit_type: DocumentType | str | None = None,
) -> tuple[DocumentType, str]:
    """Deterministically determine and validate document type and canonical MIME type.
    
    Resolution Strategy:
    1. If explicit DocumentType is supplied, validate it.
    2. Check explicit known content_type/MIME if available (and not generic octet-stream).
       - Note: For 'text/plain', checks file extension to correctly classify Markdown (.md/.markdown).
       - Unsupported explicit MIME types (e.g. image/png, text/csv) fail immediately.
    3. Fall back to file extension lookup.
    
    Args:
        content_type: Optional MIME type from source or metadata.
        filename_or_path: Optional filename or storage path.
        explicit_type: Optional explicit DocumentType or string override.
        
    Returns:
        Tuple of (detected DocumentType, resolved canonical MIME type).
        
    Raises:
        UnsupportedDocumentTypeError: If document format or MIME type is not supported.
    """
    # 1. Explicit type override
    if explicit_type is not None:
        if isinstance(explicit_type, DocumentType):
            return explicit_type, content_type or get_canonical_mime_type(explicit_type)
        if isinstance(explicit_type, str):
            clean_type = explicit_type.strip().lower()
            try:
                doc_type = DocumentType(clean_type)
                return doc_type, content_type or get_canonical_mime_type(doc_type)
            except ValueError:
                raise UnsupportedDocumentTypeError(
                    f"Unsupported explicit document type '{explicit_type}'. "
                    "Supported types are: pdf, docx, txt, markdown."
                )

    # Clean and inspect content_type
    clean_mime = ""
    if content_type and isinstance(content_type, str):
        clean_mime = content_type.split(";")[0].strip().lower()

    # 2. Inspect MIME type if provided and not generic
    if clean_mime and clean_mime not in GENERIC_MIME_TYPES:
        if clean_mime in MIME_TO_DOCUMENT_TYPE:
            matched_type = MIME_TO_DOCUMENT_TYPE[clean_mime]
            # Special case: text/plain might actually be a markdown file
            if matched_type == DocumentType.TXT and filename_or_path:
                ext = normalize_extension(filename_or_path)
                if ext in (".md", ".markdown"):
                    return DocumentType.MARKDOWN, "text/markdown"
            return matched_type, clean_mime

        # Explicit unsupported MIME type provided
        raise UnsupportedDocumentTypeError(
            f"Unsupported content type '{content_type}'. "
            "Supported document types are: PDF, DOCX, TXT, Markdown."
        )

    # 3. Extension fallback
    if filename_or_path:
        ext = normalize_extension(filename_or_path)
        if ext in EXTENSION_TO_DOCUMENT_TYPE:
            detected_type = EXTENSION_TO_DOCUMENT_TYPE[ext]
            canonical_mime = get_canonical_mime_type(detected_type)
            return detected_type, canonical_mime

        raise UnsupportedDocumentTypeError(
            f"Document '{filename_or_path}' has an unsupported file format '{ext}'. "
            "Supported formats are: .pdf, .docx, .txt, .md, .markdown"
        )

    raise UnsupportedDocumentTypeError(
        "Cannot determine document type: neither a supported content type nor filename was provided."
    )
