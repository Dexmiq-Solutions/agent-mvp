"""Abstract base class for document ingestion services."""

from abc import ABC, abstractmethod
from typing import Optional, Union

from rag.acquisition.models import SourceDocument, SourceReference
from rag.ingestion.models import DocumentSourceReference, IngestedDocument

IngestionSourceInput = Union[SourceDocument, DocumentSourceReference, SourceReference, str]


class BaseIngestionService(ABC):
    """Abstract interface defining the ingestion service contract.
    
    The ingestion layer is responsible for bringing acquired source documents
    into the application pipeline as standardized IngestedDocument objects,
    retrieving their raw contents asynchronously from object storage while
    preserving source metadata.
    
    IMPORTANT:
    This layer strictly does NOT parse, extract text, clean, chunk, embed, or index.
    """

    @abstractmethod
    async def ingest(
        self,
        source: IngestionSourceInput,
        project_id: Optional[str] = None,
    ) -> IngestedDocument:
        """Ingest a single acquired document reference into the pipeline.
        
        Args:
            source: Reference to an acquired document. May be a SourceDocument,
                DocumentSourceReference, SourceReference, or storage path string.
            project_id: Mandatory project identifier if not present in the source reference.
            
        Returns:
            Standardized IngestedDocument containing raw file bytes and preserved metadata.
            
        Raises:
            InvalidIngestionInputError: If input reference or project_id is invalid.
            UnsupportedDocumentTypeError: If document format is not supported.
            DocumentRetrievalError: If storage download fails.
            IngestionError: If an unexpected error occurs during ingestion.
        """

    @abstractmethod
    async def ingest_batch(
        self,
        sources: list[IngestionSourceInput],
        project_id: Optional[str] = None,
    ) -> list[IngestedDocument]:
        """Ingest multiple acquired document references into the pipeline.
        
        Args:
            sources: List of document references to ingest.
            project_id: Optional project identifier applied to references lacking one.
            
        Returns:
            List of successfully IngestedDocument representations.
            
        Raises:
            IngestionError: If batch ingestion fails.
        """
