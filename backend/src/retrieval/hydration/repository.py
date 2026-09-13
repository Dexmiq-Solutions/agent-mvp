"""Repository abstractions for accessing stored chunks from the PostgreSQL Content Store."""

import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from db.session import get_async_session_maker
from exceptions.retrieval import ChunkHydrationValidationError, DatabaseRetrievalError
from models.chunk import ChunkModel

logger = get_logger(__name__)

_chunk_repository: Optional["BaseChunkRepository"] = None


class BaseChunkRepository(ABC):
    """Abstract interface defining set-based Content Store retrieval."""

    @abstractmethod
    async def fetch_chunks(
        self,
        project_id: str,
        chunk_ids: Sequence[str],
        session: Optional[AsyncSession] = None,
    ) -> dict[str, ChunkModel]:
        """Fetch chunk records for the given chunk IDs scoped strictly to project_id.

        Args:
            project_id: Non-empty project/tenant ID enforcing project boundary isolation.
            chunk_ids: Sequence of chunk IDs to retrieve.
            session: Optional active AsyncSession. If omitted, repository manages session.

        Returns:
            dict[str, ChunkModel]: Mapping of chunk_id to stored ChunkModel record.
        """
        pass


class SQLAlchemyChunkRepository(BaseChunkRepository):
    """SQLAlchemy-backed implementation of ChunkRepository for PostgreSQL Content Store."""

    def __init__(
        self,
        session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    ) -> None:
        """Initialize SQLAlchemyChunkRepository.

        Args:
            session_maker: Optional async sessionmaker factory. Defaults to global session factory.
        """
        self._session_maker = session_maker

    def _get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Return the active async session maker factory."""
        if self._session_maker is not None:
            return self._session_maker
        return get_async_session_maker()

    async def _execute_batch_query(
        self,
        session: AsyncSession,
        project_id: str,
        chunk_ids: list[str],
    ) -> dict[str, ChunkModel]:
        """Execute set-based SELECT query against chunks table.

        Args:
            session: Active AsyncSession.
            project_id: Validated project ID.
            chunk_ids: List of unique non-empty chunk IDs.

        Returns:
            dict[str, ChunkModel]: Mapping of chunk_id to retrieved ChunkModel.
        """
        stmt = select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.chunk_id.in_(chunk_ids),
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        return {record.chunk_id: record for record in records}

    async def fetch_chunks(
        self,
        project_id: str,
        chunk_ids: Sequence[str],
        session: Optional[AsyncSession] = None,
    ) -> dict[str, ChunkModel]:
        """Fetch chunk records for chunk IDs scoped to project_id.

        Performs a single batched database lookup avoiding N+1 queries.

        Args:
            project_id: Tenant/project ID enforcing isolation boundary.
            chunk_ids: Sequence of candidate chunk IDs.
            session: Optional active AsyncSession.

        Returns:
            dict[str, ChunkModel]: Mapping of chunk_id to stored ChunkModel record.

        Raises:
            ChunkHydrationValidationError: If project_id or chunk_ids are invalid.
            DatabaseRetrievalError: If PostgreSQL query execution fails.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise ChunkHydrationValidationError("project_id must be a non-empty string.")
        clean_project_id = project_id.strip()

        if not isinstance(chunk_ids, Sequence) or isinstance(chunk_ids, (str, bytes)):
            raise ChunkHydrationValidationError(
                f"chunk_ids must be a Sequence, got '{type(chunk_ids).__name__}'."
            )

        # Deduplicate and filter non-empty string IDs
        unique_ids = list(dict.fromkeys(str(cid).strip() for cid in chunk_ids if cid and str(cid).strip()))
        if not unique_ids:
            return {}

        start_time = time.perf_counter()

        try:
            if session is not None:
                records_by_id = await self._execute_batch_query(session, clean_project_id, unique_ids)
            else:
                session_factory = self._get_session_maker()
                async with session_factory() as sess:
                    records_by_id = await self._execute_batch_query(sess, clean_project_id, unique_ids)
        except (ChunkHydrationValidationError, DatabaseRetrievalError):
            raise
        except SQLAlchemyError as exc:
            logger.error(
                "SQLAlchemy error fetching chunks: project_id='%s', requested=%d, error=%s",
                clean_project_id,
                len(unique_ids),
                exc,
            )
            raise DatabaseRetrievalError(
                f"Database query failed while fetching chunks for project '{clean_project_id}': {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error(
                "Unexpected error fetching chunks from PostgreSQL: project_id='%s', requested=%d, error=%s",
                clean_project_id,
                len(unique_ids),
                exc,
            )
            raise DatabaseRetrievalError(
                f"Unexpected database error fetching chunks for project '{clean_project_id}': {exc}",
                original_error=exc,
            ) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "Fetched chunk batch from PostgreSQL: project_id='%s', requested=%d, found=%d, latency_ms=%.2f",
            clean_project_id,
            len(unique_ids),
            len(records_by_id),
            elapsed_ms,
        )

        return records_by_id


def get_chunk_repository(
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    repository: Optional[BaseChunkRepository] = None,
) -> BaseChunkRepository:
    """Retrieve or initialize singleton ChunkRepository instance.

    Args:
        session_maker: Optional custom async sessionmaker.
        repository: Optional explicit BaseChunkRepository override.

    Returns:
        BaseChunkRepository: Configured repository instance.
    """
    global _chunk_repository
    if repository is not None:
        return repository

    if _chunk_repository is None or session_maker is not None:
        repo = SQLAlchemyChunkRepository(session_maker=session_maker)
        if session_maker is None:
            _chunk_repository = repo
        return repo
    return _chunk_repository


def reset_chunk_repository() -> None:
    """Reset the cached singleton ChunkRepository instance (useful for tests)."""
    global _chunk_repository
    _chunk_repository = None
