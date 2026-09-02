"""Qdrant vector store implementation."""

from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.vector import (
    CollectionConfigurationError,
    CollectionNotFoundError,
    VectorDeletionError,
    VectorInputValidationError,
    VectorSearchError,
    VectorStoreAuthenticationError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from storage.vector.base import BaseVectorStore
from storage.vector.client import get_async_qdrant_client
from storage.vector.models import VectorPayload, VectorRecord, VectorSearchResult

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation of the BaseVectorStore interface.
    
    Provides asynchronous vector storage, idempotent collection management, batch upserts,
    deletions, and similarity search with mandatory project-level isolation.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        client: Optional[AsyncQdrantClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Qdrant vector store.
        
        Args:
            collection_name: Optional collection name override. Defaults to settings.QDRANT_COLLECTION_NAME.
            client: Pre-configured AsyncQdrantClient instance. If None, resolved from client provider.
            settings: Application settings. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._collection_name = collection_name or self._settings.QDRANT_COLLECTION_NAME
        if not self._collection_name:
            raise VectorStoreConfigurationError(
                "Qdrant collection name must be configured via QDRANT_COLLECTION_NAME."
            )
        self._client = client

    @property
    def collection_name(self) -> str:
        """Return the target collection name."""
        return self._collection_name

    def _get_client(self) -> AsyncQdrantClient:
        """Resolve the active asynchronous Qdrant client."""
        if self._client is not None:
            return self._client
        return get_async_qdrant_client(self._settings)

    # --------------------------------------------------------------------------
    # Validation & Helper Methods
    # --------------------------------------------------------------------------

    def _validate_project_id(self, project_id: Any) -> str:
        """Validate that project_id is provided and non-empty for tenant isolation."""
        if not isinstance(project_id, str) or not project_id.strip():
            raise VectorInputValidationError(
                "project_id must be a non-empty string to enforce project isolation."
            )
        return project_id.strip()

    def _validate_records(self, records: Any) -> list[VectorRecord]:
        """Validate batch vector records synchronously before network transmission."""
        if not isinstance(records, (list, tuple)):
            raise VectorInputValidationError(
                f"Records must be a list or tuple of VectorRecord instances, got {type(records).__name__}."
            )
        if len(records) == 0:
            raise VectorInputValidationError("Vector records batch cannot be empty.")

        validated: list[VectorRecord] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, VectorRecord):
                raise VectorInputValidationError(
                    f"Item at index {i} must be a VectorRecord instance, got {type(rec).__name__}."
                )
            if rec.id is None or (isinstance(rec.id, str) and not rec.id.strip()):
                raise VectorInputValidationError(f"Record at index {i} has an invalid or empty ID.")
            if not isinstance(rec.vector, (list, tuple)) or len(rec.vector) == 0:
                raise VectorInputValidationError(
                    f"Record at index {i} has an empty or invalid vector."
                )
            if not isinstance(rec.payload, VectorPayload):
                raise VectorInputValidationError(
                    f"Record at index {i} payload must be a VectorPayload instance."
                )
            self._validate_project_id(rec.payload.project_id)
            if not rec.payload.document_id or not rec.payload.document_id.strip():
                raise VectorInputValidationError(
                    f"Record at index {i} payload is missing document_id."
                )
            if not rec.payload.chunk_id or not rec.payload.chunk_id.strip():
                raise VectorInputValidationError(f"Record at index {i} payload is missing chunk_id.")
            validated.append(rec)

        return validated

    def _build_filter(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        document_version_id: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> models.Filter:
        """Build a Qdrant Filter with mandatory project isolation."""
        clean_project_id = self._validate_project_id(project_id)
        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="project_id",
                match=models.MatchValue(value=clean_project_id),
            )
        ]

        if document_id is not None and document_id.strip():
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id.strip()),
                )
            )

        if document_version_id is not None and document_version_id.strip():
            must_conditions.append(
                models.FieldCondition(
                    key="document_version_id",
                    match=models.MatchValue(value=document_version_id.strip()),
                )
            )

        if filter_metadata:
            for key, val in filter_metadata.items():
                if val is not None:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val),
                        )
                    )

        return models.Filter(must=must_conditions)

    # --------------------------------------------------------------------------
    # Collection Management
    # --------------------------------------------------------------------------

    async def ensure_collection_exists(
        self,
        vector_size: Optional[int] = None,
        distance: str = "Cosine",
    ) -> bool:
        """Verify that the target collection exists, creating it idempotently if missing."""
        client = self._get_client()
        target_size = vector_size or self._settings.QDRANT_VECTOR_SIZE

        try:
            exists = await client.collection_exists(self._collection_name)
            if exists:
                collection_info = await client.get_collection(self._collection_name)
                # Verify vector dimensionality if available in collection info
                vectors_config = getattr(collection_info.config.params, "vectors", None)
                if isinstance(vectors_config, models.VectorParams):
                    existing_size = vectors_config.size
                    if target_size is not None and existing_size != target_size:
                        raise CollectionConfigurationError(
                            f"Collection '{self._collection_name}' already exists with vector size "
                            f"{existing_size}, but requested size is {target_size}. "
                            f"Aborting without destructive overwrite."
                        )
                logger.debug(
                    "Collection '%s' already exists and is compatible", self._collection_name
                )
                return False

            distance_enum = getattr(models.Distance, distance.upper(), models.Distance.COSINE)
            logger.info(
                "Creating Qdrant collection '%s' (vector_size: %d, distance: %s)",
                self._collection_name,
                target_size,
                distance,
            )
            await client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(size=target_size, distance=distance_enum),
            )
            return True
        except CollectionConfigurationError:
            raise
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreAuthenticationError(
                    "Authentication failed with Qdrant.", original_error=exc
                ) from exc
            raise VectorStoreError(
                f"Failed to check or create collection '{self._collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            if isinstance(exc, (VectorStoreError, VectorStoreConfigurationError)):
                raise
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant for collection verification: {exc}",
                original_error=exc,
            ) from exc

    # --------------------------------------------------------------------------
    # Upsert Operations
    # --------------------------------------------------------------------------

    async def upsert(self, records: list[VectorRecord]) -> int:
        """Upsert a batch of vector records into Qdrant."""
        valid_records = self._validate_records(records)
        client = self._get_client()

        points: list[models.PointStruct] = [
            models.PointStruct(
                id=rec.id,
                vector=rec.vector,
                payload=rec.payload.to_dict(),
            )
            for rec in valid_records
        ]

        try:
            logger.debug(
                "Upserting %d vector points into collection '%s'",
                len(points),
                self._collection_name,
            )
            await client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
            return len(points)
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreAuthenticationError(
                    "Authentication failed with Qdrant.", original_error=exc
                ) from exc
            if exc.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{self._collection_name}' not found.", original_error=exc
                ) from exc
            raise VectorUpsertError(
                f"Failed to upsert points into collection '{self._collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            if isinstance(exc, (VectorStoreError, VectorInputValidationError)):
                raise
            raise VectorUpsertError(
                f"Unexpected error upserting vector points: {exc}", original_error=exc
            ) from exc

    # --------------------------------------------------------------------------
    # Deletion Operations
    # --------------------------------------------------------------------------

    async def delete(self, point_ids: list[str | int]) -> int:
        """Delete specific vector points by ID."""
        if not isinstance(point_ids, (list, tuple)) or len(point_ids) == 0:
            raise VectorInputValidationError("point_ids list cannot be empty.")

        client = self._get_client()
        try:
            logger.debug(
                "Deleting %d points from collection '%s'",
                len(point_ids),
                self._collection_name,
            )
            await client.delete(
                collection_name=self._collection_name,
                points_selector=point_ids,
                wait=True,
            )
            return len(point_ids)
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreAuthenticationError(
                    "Authentication failed with Qdrant.", original_error=exc
                ) from exc
            if exc.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{self._collection_name}' not found.", original_error=exc
                ) from exc
            raise VectorDeletionError(
                f"Failed to delete points from collection '{self._collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            if isinstance(exc, (VectorStoreError, VectorInputValidationError)):
                raise
            raise VectorDeletionError(
                f"Unexpected error deleting vector points: {exc}", original_error=exc
            ) from exc

    async def delete_by_filter(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        document_version_id: Optional[str] = None,
    ) -> bool:
        """Delete vector points matching project and document filters."""
        query_filter = self._build_filter(
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )
        client = self._get_client()

        try:
            logger.debug(
                "Deleting points by filter in collection '%s' (project_id: %s, document_id: %s)",
                self._collection_name,
                project_id,
                document_id,
            )
            await client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            )
            return True
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreAuthenticationError(
                    "Authentication failed with Qdrant.", original_error=exc
                ) from exc
            if exc.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{self._collection_name}' not found.", original_error=exc
                ) from exc
            raise VectorDeletionError(
                f"Failed to delete points by filter in collection '{self._collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            if isinstance(exc, (VectorStoreError, VectorInputValidationError)):
                raise
            raise VectorDeletionError(
                f"Unexpected error deleting points by filter: {exc}", original_error=exc
            ) from exc

    # --------------------------------------------------------------------------
    # Similarity Search Operations
    # --------------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        project_id: str,
        limit: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> list[VectorSearchResult]:
        """Perform similarity search with mandatory project isolation."""
        if not isinstance(query_vector, (list, tuple)) or len(query_vector) == 0:
            raise VectorInputValidationError("query_vector must be a non-empty list of floats.")
        if limit <= 0:
            raise VectorInputValidationError(f"limit must be a positive integer, got {limit}.")

        query_filter = self._build_filter(
            project_id=project_id,
            filter_metadata=filter_metadata,
        )
        client = self._get_client()

        try:
            logger.debug(
                "Executing vector search in '%s' for project '%s' (limit: %d)",
                self._collection_name,
                project_id,
                limit,
            )
            response = await client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
            )

            results: list[VectorSearchResult] = []
            for pt in response.points:
                payload = VectorPayload.from_dict(pt.payload or {})
                # Double-check project isolation consistency
                if payload.project_id != project_id:
                    logger.warning(
                        "Encountered mismatched project_id '%s' in search result for project '%s'",
                        payload.project_id,
                        project_id,
                    )
                    continue

                raw_vector: Optional[list[float]] = None
                if with_vectors and pt.vector is not None:
                    if isinstance(pt.vector, list):
                        raw_vector = pt.vector

                results.append(
                    VectorSearchResult(
                        id=pt.id,
                        score=float(pt.score),
                        payload=payload,
                        vector=raw_vector,
                    )
                )

            logger.debug("Found %d matching vector points", len(results))
            return results
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreAuthenticationError(
                    "Authentication failed with Qdrant.", original_error=exc
                ) from exc
            if exc.status_code == 404:
                raise CollectionNotFoundError(
                    f"Collection '{self._collection_name}' not found.", original_error=exc
                ) from exc
            raise VectorSearchError(
                f"Failed to execute vector search in collection '{self._collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            if isinstance(exc, (VectorStoreError, VectorInputValidationError)):
                raise
            raise VectorSearchError(
                f"Unexpected error executing vector search: {exc}", original_error=exc
            ) from exc
