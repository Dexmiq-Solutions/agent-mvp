"""Abstract base interface for RAG document processing and indexing pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseDocumentProcessingPipeline(ABC):
    """Abstract interface defining the contract for document processing pipelines.

    Decouples the application-level Document Processing Lifecycle from the concrete
    implementation of RAG stages (acquisition, ingestion, parsing, cleaning,
    normalization, chunking, enrichment, embeddings, sparse representations, and indexing).
    """

    @abstractmethod
    async def process_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        storage_bucket: str,
        storage_path: str,
        original_filename: str,
        content_type: Optional[str] = None,
    ) -> Any:
        """Execute processing for a verified document version.

        Args:
            project_id: Verified owning project identifier.
            document_id: Verified owning document identifier.
            document_version_id: Verified physical document version identifier.
            storage_bucket: Object storage bucket containing the raw document.
            storage_path: Relative storage path of the raw document file.
            original_filename: Original name of the uploaded document file.
            content_type: Optional MIME content type.

        Returns:
            Pipeline execution report or result data structure.

        Raises:
            Exception: If pipeline processing fails at any stage.
        """
