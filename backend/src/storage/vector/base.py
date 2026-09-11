"""Abstract base class defining the vector store interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from storage.vector.models import VectorRecord, VectorSearchResult


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and similarity search.
    
    Decouples the application and future RAG pipelines from specific vector store vendor SDKs
    (such as Qdrant).
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Return the target collection name."""

    @abstractmethod
    async def ensure_collection_exists(
        self,
        vector_size: Optional[int] = None,
        distance: str = "Cosine",
    ) -> bool:
        """Verify that the target collection exists, creating it idempotently if missing.
        
        Args:
            vector_size: Dimension of vector embeddings (e.g., 1024). If None, uses configured default.
            distance: Distance metric ('Cosine', 'Euclid', 'Dot').
            
        Returns:
            bool: True if collection was created, False if it already existed.
            
        Raises:
            CollectionConfigurationError: If collection exists with conflicting dimensions.
            VectorStoreError: If collection creation fails.
        """

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> int:
        """Upsert one or more vector records in a batch.
        
        Args:
            records: List of VectorRecord instances containing ID, vector, and VectorPayload.
            
        Returns:
            int: Number of records successfully upserted.
            
        Raises:
            VectorInputValidationError: If records batch is empty or malformed.
            VectorUpsertError: If upsert operation fails.
        """

    @abstractmethod
    async def delete(self, point_ids: list[str | int]) -> int:
        """Delete specific vector points by their IDs.
        
        Args:
            point_ids: List of point IDs (UUID string or integer) to remove.
            
        Returns:
            int: Number of point IDs requested for deletion.
            
        Raises:
            VectorInputValidationError: If point_ids is empty.
            VectorDeletionError: If deletion fails.
        """

    @abstractmethod
    async def delete_by_filter(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        document_version_id: Optional[str] = None,
    ) -> bool:
        """Delete vector points matching project and optional document/version filters.
        
        Enforces project isolation so deletions are strictly scoped to the specified project.
        
        Args:
            project_id: Mandatory project ID for tenant isolation.
            document_id: Optional document ID to delete all chunks belonging to a document.
            document_version_id: Optional document version ID.
            
        Returns:
            bool: True if deletion request succeeded.
            
        Raises:
            VectorInputValidationError: If project_id is empty or invalid.
            VectorDeletionError: If deletion fails.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        project_id: str,
        limit: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search with mandatory project isolation.
        
        Args:
            query_vector: Query embedding vector.
            project_id: Mandatory project ID to strictly enforce tenant isolation.
            limit: Maximum number of top-k results to return.
            filter_metadata: Optional additional metadata key-value filters.
            score_threshold: Optional minimum similarity score threshold.
            with_vectors: Whether to include raw vectors in search results.
            
        Returns:
            list[VectorSearchResult]: List of ranked search results.
            
        Raises:
            VectorInputValidationError: If query_vector is empty or project_id is missing.
            VectorSearchError: If search query fails.
        """

    async def search_batch(
        self,
        query_vectors: list[list[float]],
        project_id: str,
        limit: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> list[list[VectorSearchResult]]:
        """Perform batch vector similarity search for multiple query vectors with project isolation.

        Default fallback implementation executes search() sequentially if the underlying
        vector store does not provide a native batch search operation.

        Args:
            query_vectors: List of query embedding vectors to search.
            project_id: Mandatory project ID to strictly enforce tenant isolation.
            limit: Maximum number of top-k results to return per query vector.
            filter_metadata: Optional additional metadata key-value filters.
            score_threshold: Optional minimum similarity score threshold.
            with_vectors: Whether to include raw vectors in search results.

        Returns:
            list[list[VectorSearchResult]]: List of ranked search result lists corresponding
                1-to-1 with input query vectors.

        Raises:
            VectorInputValidationError: If query_vectors is empty, contains empty vectors, or project_id is invalid.
            VectorSearchError: If search query fails.
        """
        results: list[list[VectorSearchResult]] = []
        for qv in query_vectors:
            res = await self.search(
                query_vector=qv,
                project_id=project_id,
                limit=limit,
                filter_metadata=filter_metadata,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
            )
            results.append(res)
        return results
