"""Chunk persistence service for the PostgreSQL Content Store."""

from collections.abc import Sequence
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from db.session import get_async_session_maker
from exceptions.indexing import (
    IndexingOperationError,
    InvalidIndexingInputError,
)
from models.chunk import ChunkModel

logger = get_logger(__name__)

_chunk_persistence_service: Optional["ChunkPersistenceService"] = None


class ChunkPersistenceService:
    """Service responsible for persisting generated chunks into the PostgreSQL Content Store.

    Ensures:
      - Atomic persistence of chunk content and metadata
      - Full preservation of structural hierarchy and coordinates
      - Strict tenant isolation enforced at database boundary
      - Idempotent cleanup and replacement on re-indexing
      - Short transaction boundary avoiding open connections during external calls
    """

    def __init__(
        self,
        session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize ChunkPersistenceService.

        Args:
            session_maker: Optional async session maker factory.
            settings: Optional application settings.
        """
        self._session_maker = session_maker
        self._settings = settings or get_settings()

    def _get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Resolve active async sessionmaker factory."""
        if self._session_maker is not None:
            return self._session_maker
        return get_async_session_maker(settings=self._settings)

    @staticmethod
    def _extract_chunk_metadata(chunk: Any) -> dict[str, Any]:
        """Extract a clean, serializable metadata dictionary from a chunk."""
        if hasattr(chunk, "enriched_metadata") and chunk.enriched_metadata is not None:
            if hasattr(chunk.enriched_metadata, "to_dict") and callable(chunk.enriched_metadata.to_dict):
                return chunk.enriched_metadata.to_dict()
        if hasattr(chunk, "metadata") and chunk.metadata is not None:
            if hasattr(chunk.metadata, "to_dict") and callable(chunk.metadata.to_dict):
                return chunk.metadata.to_dict()
            if isinstance(chunk.metadata, dict):
                return dict(chunk.metadata)
        return {}

    def _build_chunk_model(
        self,
        chunk: Any,
        project_id: str,
        document_id: str,
        document_version_id: str,
        index: int,
    ) -> ChunkModel:
        """Construct a ChunkModel entity from an in-memory chunk."""
        chunk_id = getattr(chunk, "chunk_id", None) or f"{document_id}_chunk_{index}"
        content = getattr(chunk, "content", "")
        if not content or not content.strip():
            raise InvalidIndexingInputError(f"Chunk '{chunk_id}' has empty content.")

        # Extract contextual content if enriched
        contextual_content: Optional[str] = None
        if getattr(chunk, "is_contextually_enriched", False) and getattr(chunk, "context_text", None):
            contextual_content = getattr(chunk, "contextual_content", None) or f"{chunk.context_text}\n\n{content}"

        # Hierarchy & structural positioning
        heading = getattr(chunk, "heading", None)
        heading_level = getattr(chunk, "heading_level", None)
        raw_section_path = getattr(chunk, "section_path", None)
        section_path = list(raw_section_path) if raw_section_path else []

        parent_element_id = getattr(chunk, "parent_element_id", None)
        parent_chunk_id = getattr(chunk, "parent_chunk_id", None)

        raw_element_types = getattr(chunk, "element_types", None)
        if raw_element_types:
            element_types = [
                et.value if hasattr(et, "value") else str(et)
                for et in raw_element_types
            ]
        else:
            element_types = []

        chunk_index = getattr(chunk, "index", index)
        meta = self._extract_chunk_metadata(chunk)

        return ChunkModel(
            chunk_id=chunk_id,
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
            content=content,
            contextual_content=contextual_content,
            chunk_index=chunk_index,
            heading=heading,
            heading_level=heading_level,
            section_path=section_path,
            parent_element_id=parent_element_id,
            parent_chunk_id=parent_chunk_id,
            element_types=element_types,
            chunk_metadata=meta,
        )

    async def persist_chunks(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        chunks: Sequence[Any],
        session: Optional[AsyncSession] = None,
    ) -> list[ChunkModel]:
        """Persist chunks atomically into PostgreSQL Content Store with idempotency.

        Args:
            project_id: Owning project identifier enforcing isolation.
            document_id: Source document identifier.
            document_version_id: Target physical document version identifier.
            chunks: Sequence of chunk objects (DocumentChunk, EnrichedChunk, etc.).
            session: Optional active AsyncSession. If None, manages own short transaction.

        Returns:
            list[ChunkModel]: List of persisted ChunkModel records.

        Raises:
            InvalidIndexingInputError: If inputs or chunk records are invalid.
            IndexingOperationError: If database persistence fails.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise InvalidIndexingInputError("project_id must be a non-empty string.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidIndexingInputError("document_id must be a non-empty string.")
        if not isinstance(document_version_id, str) or not document_version_id.strip():
            raise InvalidIndexingInputError("document_version_id must be a non-empty string.")
        if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
            raise InvalidIndexingInputError("chunks must be a non-empty sequence.")

        clean_project_id = project_id.strip()
        clean_document_id = document_id.strip()
        clean_version_id = document_version_id.strip()

        # Build models and enforce tenant isolation across chunks
        chunk_models: list[ChunkModel] = []
        for i, chk in enumerate(chunks):
            chk_proj = getattr(chk, "project_id", clean_project_id)
            if chk_proj != clean_project_id:
                raise InvalidIndexingInputError(
                    f"Project isolation violation: Chunk {i} has project_id '{chk_proj}', "
                    f"expected '{clean_project_id}'."
                )
            chk_doc = getattr(chk, "document_id", clean_document_id)
            if chk_doc != clean_document_id:
                raise InvalidIndexingInputError(
                    f"Document isolation violation: Chunk {i} has document_id '{chk_doc}', "
                    f"expected '{clean_document_id}'."
                )
            model = self._build_chunk_model(
                chunk=chk,
                project_id=clean_project_id,
                document_id=clean_document_id,
                document_version_id=clean_version_id,
                index=i,
            )
            chunk_models.append(model)

        logger.info(
            "Persisting %d chunks to PostgreSQL for doc '%s' (ver: '%s', project: '%s')",
            len(chunk_models),
            clean_document_id,
            clean_version_id,
            clean_project_id,
        )

        async def _execute_persistence(s: AsyncSession) -> None:
            # 1. Clean up existing chunks for this document to prevent PK collisions and ensure idempotency
            delete_stmt = delete(ChunkModel).where(
                ChunkModel.project_id == clean_project_id,
                ChunkModel.document_id == clean_document_id,
            )
            await s.execute(delete_stmt)

            # 2. Add all new chunk models
            s.add_all(chunk_models)
            await s.flush()

        try:
            if session is not None:
                await _execute_persistence(session)
            else:
                maker = self._get_session_maker()
                async with maker() as new_session:
                    async with new_session.begin():
                        await _execute_persistence(new_session)

            logger.info(
                "Successfully persisted %d chunks to PostgreSQL for version '%s'",
                len(chunk_models),
                clean_version_id,
            )
            return chunk_models
        except InvalidIndexingInputError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to persist chunks in PostgreSQL for version '%s': %s",
                clean_version_id,
                exc,
                exc_info=True,
            )
            raise IndexingOperationError(
                f"Failed to persist chunks into PostgreSQL for version '{clean_version_id}': {exc}",
                original_error=exc,
            ) from exc

    async def delete_chunks_for_version(
        self,
        project_id: str,
        document_version_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """Delete all chunks for a specific document version from PostgreSQL."""
        if not project_id or not document_version_id:
            raise InvalidIndexingInputError("project_id and document_version_id must be non-empty.")

        clean_project_id = project_id.strip()
        clean_version_id = document_version_id.strip()

        async def _execute_delete(s: AsyncSession) -> int:
            stmt = delete(ChunkModel).where(
                ChunkModel.project_id == clean_project_id,
                ChunkModel.document_version_id == clean_version_id,
            )
            res = await s.execute(stmt)
            return getattr(res, "rowcount", 0)

        try:
            if session is not None:
                return await _execute_delete(session)
            else:
                maker = self._get_session_maker()
                async with maker() as new_session:
                    async with new_session.begin():
                        return await _execute_delete(new_session)
        except Exception as exc:
            logger.error(
                "Failed to delete chunks for version '%s' in project '%s': %s",
                clean_version_id,
                clean_project_id,
                exc,
            )
            raise IndexingOperationError(
                f"Failed to delete chunks for version '{clean_version_id}': {exc}",
                original_error=exc,
            ) from exc


def get_chunk_persistence_service(
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    settings: Optional[Settings] = None,
) -> ChunkPersistenceService:
    """Get or create the singleton ChunkPersistenceService instance."""
    global _chunk_persistence_service
    if session_maker is not None:
        return ChunkPersistenceService(session_maker=session_maker, settings=settings)

    if _chunk_persistence_service is None:
        _chunk_persistence_service = ChunkPersistenceService(settings=settings)
    return _chunk_persistence_service


def reset_chunk_persistence_service() -> None:
    """Reset the cached default chunk persistence service instance. Useful for tests."""
    global _chunk_persistence_service
    _chunk_persistence_service = None
