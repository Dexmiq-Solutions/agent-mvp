"""Abstract base class for source document acquisition services."""

from abc import ABC, abstractmethod
from typing import Optional

from acquisition.models import AcquisitionResult, SourceDocument


class BaseAcquisitionService(ABC):
    """Abstract interface defining the acquisition service contract.
    
    The acquisition layer is responsible exclusively for discovering available
    source documents for a project and producing structured references for downstream
    ingestion, without downloading file bodies, parsing, chunking, or indexing.
    """

    @abstractmethod
    async def acquire(
        self,
        project_id: str,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AcquisitionResult:
        """Discover and acquire all available supported source documents for a project.
        
        Args:
            project_id: Mandatory project ID ensuring strict project isolation.
            prefix: Optional sub-folder path within the project namespace.
            limit: Optional upper bound on the number of documents to acquire.
            
        Returns:
            AcquisitionResult containing acquired SourceDocument references and skipped items.
            
        Raises:
            InvalidSourceReferenceError: If project_id is empty or invalid.
            SourceDiscoveryError: If discovery fails in the underlying storage provider.
            AcquisitionConfigurationError: If acquisition service is improperly configured.
        """

    @abstractmethod
    async def acquire_document(
        self,
        project_id: str,
        storage_path: str,
    ) -> SourceDocument:
        """Acquire and validate a single specific source document by its storage path.
        
        Args:
            project_id: Mandatory project ID ensuring strict project isolation.
            storage_path: Relative storage path of the target document.
            
        Returns:
            Validated SourceDocument reference.
            
        Raises:
            InvalidSourceReferenceError: If project_id or path is invalid or attempts cross-tenant access.
            UnsupportedDocumentError: If the document format is not supported.
            SourceDiscoveryError: If the object cannot be found or metadata retrieval fails.
        """
