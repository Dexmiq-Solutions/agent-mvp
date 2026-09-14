"""Single source of truth for supported document formats and content types."""

from enum import Enum
from pathlib import PurePosixPath
from typing import NamedTuple


class DocumentType(str, Enum):
    """Enumeration of all supported document types for the indexing pipeline."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"


class FormatSpec(NamedTuple):
    """Specification metadata defining a supported document format."""

    document_type: DocumentType
    primary_extension: str
    supported_extensions: tuple[str, ...]
    canonical_mime_type: str
    supported_mime_types: tuple[str, ...]


SUPPORTED_FORMAT_SPECS: dict[DocumentType, FormatSpec] = {
    DocumentType.PDF: FormatSpec(
        document_type=DocumentType.PDF,
        primary_extension=".pdf",
        supported_extensions=(".pdf",),
        canonical_mime_type="application/pdf",
        supported_mime_types=("application/pdf",),
    ),
    DocumentType.DOCX: FormatSpec(
        document_type=DocumentType.DOCX,
        primary_extension=".docx",
        supported_extensions=(".docx",),
        canonical_mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        supported_mime_types=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ),
    ),
    DocumentType.TXT: FormatSpec(
        document_type=DocumentType.TXT,
        primary_extension=".txt",
        supported_extensions=(".txt",),
        canonical_mime_type="text/plain",
        supported_mime_types=("text/plain",),
    ),
    DocumentType.MARKDOWN: FormatSpec(
        document_type=DocumentType.MARKDOWN,
        primary_extension=".md",
        supported_extensions=(".md", ".markdown"),
        canonical_mime_type="text/markdown",
        supported_mime_types=(
            "text/markdown",
            "text/x-markdown",
            "text/plain",
        ),
    ),
}

# Derived extension lookup dictionary for O(1) matching:
EXTENSION_TO_DOCUMENT_TYPE: dict[str, DocumentType] = {
    ext.lower(): doc_type
    for doc_type, spec in SUPPORTED_FORMAT_SPECS.items()
    for ext in spec.supported_extensions
}

# Supported extensions tuple for fast checking & display:
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(EXTENSION_TO_DOCUMENT_TYPE.keys()))


def normalize_extension(path_or_extension: str) -> str:
    """Extract and normalize file extension with leading dot in lowercase.
    
    Args:
        path_or_extension: File path, file name, or bare extension (e.g. 'doc.PDF', '.docx', 'md').
        
    Returns:
        Normalized extension string starting with a dot, e.g. '.pdf'.
    """
    if not path_or_extension:
        return ""
    clean = path_or_extension.strip().lower()
    if "/" in clean or "\\" in clean or "." in clean:
        suffix = PurePosixPath(clean.replace("\\", "/")).suffix
        if suffix:
            return suffix.lower()
    if not clean.startswith("."):
        clean = f".{clean}"
    return clean


def is_supported_extension(path_or_extension: str) -> bool:
    """Check if the given path, filename, or extension is a supported document format.
    
    Args:
        path_or_extension: File path, file name, or extension to check.
        
    Returns:
        True if the format is supported, False otherwise.
    """
    ext = normalize_extension(path_or_extension)
    return ext in EXTENSION_TO_DOCUMENT_TYPE


def get_document_type(path_or_extension: str) -> DocumentType | None:
    """Resolve the canonical DocumentType for a given path, filename, or extension.
    
    Args:
        path_or_extension: File path, file name, or extension.
        
    Returns:
        Matching DocumentType enum or None if unsupported.
    """
    ext = normalize_extension(path_or_extension)
    return EXTENSION_TO_DOCUMENT_TYPE.get(ext)


def get_canonical_mime_type(document_type: DocumentType) -> str:
    """Retrieve the standard MIME type for a supported DocumentType.
    
    Args:
        document_type: The DocumentType to look up.
        
    Returns:
        Canonical MIME type string.
    """
    spec = SUPPORTED_FORMAT_SPECS.get(document_type)
    if spec:
        return spec.canonical_mime_type
    return "application/octet-stream"
