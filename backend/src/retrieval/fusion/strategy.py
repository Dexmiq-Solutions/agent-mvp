"""Fusion strategies combining dense and sparse retrieval candidate lists into a unified ranking."""

from abc import ABC, abstractmethod
import math
from typing import Any, Optional, Sequence

from app.core.logging import get_logger
from exceptions.retrieval import FusionError
from retrieval.models import FusedCandidate, KeywordSearchCandidate, VectorSearchCandidate

logger = get_logger(__name__)


_DEFAULT_TOP_K = object()


class BaseFusionStrategy(ABC):
    """Abstract base class for hybrid retrieval fusion strategies.

    Answers: 'How should heterogeneous candidate rankings from dense and sparse retrieval
    be combined into a single, deduplicated, ordered candidate ranking?'
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the unique strategy identifier."""

    @abstractmethod
    def fuse(
        self,
        dense_results: Sequence[VectorSearchCandidate],
        sparse_results: Sequence[KeywordSearchCandidate],
        top_k: Any = _DEFAULT_TOP_K,
        project_id: Optional[str] = None,
    ) -> list[FusedCandidate]:
        """Combine dense and sparse candidate lists into a unified ranking.

        Args:
            dense_results: Ordered sequence of candidates retrieved via dense vector search.
            sparse_results: Ordered sequence of candidates retrieved via sparse/keyword search.
            top_k: Optional maximum number of fused candidates to return.
            project_id: Optional project/tenant ID to enforce project isolation.

        Returns:
            list[FusedCandidate]: Deduplicated, scored, and ranked candidates.

        Raises:
            FusionError: If inputs are invalid or fusion calculation fails.
        """


class ReciprocalRankFusionStrategy(BaseFusionStrategy):
    """Fusion strategy implementing Reciprocal Rank Fusion (RRF).

    Combines multiple heterogeneous candidate result lists using 1-based ranks:
        RRF(d) = sum(1.0 / (k + rank_m(d)))

    Where:
        - d is a candidate chunk identified by its stable chunk_id.
        - rank_m(d) is the 1-based rank (1, 2, ...) of chunk d in retrieval list m.
        - k is the configurable smoothing constant (conventionally 60).
    """

    def __init__(
        self,
        rrf_k: int = 60,
        default_top_k: Optional[int] = 10,
    ) -> None:
        """Initialize ReciprocalRankFusionStrategy.

        Args:
            rrf_k: Smoothing constant k added to ranks. Must be a positive integer.
            default_top_k: Default maximum number of unified candidates to return.
        """
        if not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}.")
        if default_top_k is not None and (not isinstance(default_top_k, int) or default_top_k <= 0):
            raise ValueError(f"default_top_k must be a positive integer or None, got {default_top_k}.")

        self._rrf_k = rrf_k
        self._default_top_k = default_top_k

    @property
    def strategy_name(self) -> str:
        """Return strategy identifier."""
        return "rrf"

    @property
    def rrf_k(self) -> int:
        """Return the active RRF constant k."""
        return self._rrf_k

    @property
    def default_top_k(self) -> Optional[int]:
        """Return default top-k output limit."""
        return self._default_top_k

    def fuse(
        self,
        dense_results: Sequence[VectorSearchCandidate],
        sparse_results: Sequence[KeywordSearchCandidate],
        top_k: Any = _DEFAULT_TOP_K,
        project_id: Optional[str] = None,
    ) -> list[FusedCandidate]:
        """Fuse dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: Ordered sequence of dense vector search candidates.
            sparse_results: Ordered sequence of sparse/keyword search candidates.
            top_k: Maximum number of fused candidates to return. If None, returns all unique candidates.
                   If omitted, defaults to default_top_k.
            project_id: Optional project ID to defensively enforce tenant isolation.

        Returns:
            list[FusedCandidate]: Deduplicated, scored, and ranked candidate chunks.

        Raises:
            FusionError: If inputs are invalid or fusion calculation encounters an error.
        """
        # 1. Validate Input Sequences
        if not isinstance(dense_results, Sequence) or isinstance(dense_results, (str, bytes)):
            raise FusionError(
                f"Expected Sequence for dense_results, got '{type(dense_results).__name__}'."
            )
        if not isinstance(sparse_results, Sequence) or isinstance(sparse_results, (str, bytes)):
            raise FusionError(
                f"Expected Sequence for sparse_results, got '{type(sparse_results).__name__}'."
            )

        # 2. Validate Project Isolation Parameter
        clean_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise FusionError("project_id must be a non-empty string when specified.")
            clean_project_id = project_id.strip()

        # 3. Resolve Effective Top-K Parameter
        effective_top_k = self._default_top_k if top_k is _DEFAULT_TOP_K else top_k
        if effective_top_k is not None:
            if not isinstance(effective_top_k, int) or effective_top_k <= 0:
                raise FusionError(
                    f"top_k must be a positive integer or None, got {effective_top_k}."
                )

        # 4. Hash-based candidate accumulation: O(D + S)
        # Key: chunk_id -> dict storing accumulated score and branch metadata
        accumulator: dict[str, dict[str, Any]] = {}
        discarded_count = 0

        # --- Process Dense Results ---
        seen_dense_chunks: set[str] = set()
        for rank, candidate in enumerate(dense_results, start=1):
            if candidate is None:
                discarded_count += 1
                logger.warning("Discarded null candidate at rank %d in dense results", rank)
                continue

            chunk_id = getattr(candidate, "chunk_id", None)
            document_id = getattr(candidate, "document_id", None)
            cand_project_id = getattr(candidate, "project_id", None)
            raw_score = getattr(candidate, "score", None)

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed dense candidate at rank %d: reason='missing or empty chunk_id'",
                    rank,
                )
                continue

            if not isinstance(document_id, str) or not document_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed dense candidate at rank %d: reason='missing or empty document_id'",
                    rank,
                )
                continue

            if not isinstance(cand_project_id, str) or not cand_project_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed dense candidate at rank %d: reason='missing or empty project_id'",
                    rank,
                )
                continue

            # Defensive project isolation check
            if clean_project_id is not None and cand_project_id.strip() != clean_project_id:
                discarded_count += 1
                logger.warning(
                    "Discarded leaked dense candidate at rank %d: reason='project_id mismatch' "
                    "(expected='%s', actual='%s')",
                    rank,
                    clean_project_id,
                    cand_project_id,
                )
                continue

            cid = chunk_id.strip()
            # If candidate already appeared in dense list, skip to preserve best/first rank
            if cid in seen_dense_chunks:
                continue
            seen_dense_chunks.add(cid)

            # Calculate RRF score contribution using 1-based rank
            score_contrib = 1.0 / (self._rrf_k + rank)
            doc_version = (
                candidate.document_version_id.strip()
                if (
                    getattr(candidate, "document_version_id", None)
                    and candidate.document_version_id.strip()
                )
                else None
            )

            dense_score_val = (
                float(raw_score)
                if (isinstance(raw_score, (int, float)) and math.isfinite(raw_score))
                else None
            )

            cand_metadata = dict(getattr(candidate, "metadata", None) or {})

            accumulator[cid] = {
                "chunk_id": cid,
                "document_id": document_id.strip(),
                "project_id": cand_project_id.strip(),
                "document_version_id": doc_version,
                "score": score_contrib,
                "dense_rank": rank,
                "dense_score": dense_score_val,
                "sparse_rank": None,
                "sparse_score": None,
                "metadata": cand_metadata,
            }

        # --- Process Sparse Results ---
        seen_sparse_chunks: set[str] = set()
        for rank, candidate in enumerate(sparse_results, start=1):
            if candidate is None:
                discarded_count += 1
                logger.warning("Discarded null candidate at rank %d in sparse results", rank)
                continue

            chunk_id = getattr(candidate, "chunk_id", None)
            document_id = getattr(candidate, "document_id", None)
            cand_project_id = getattr(candidate, "project_id", None)
            raw_score = getattr(candidate, "score", None)

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed sparse candidate at rank %d: reason='missing or empty chunk_id'",
                    rank,
                )
                continue

            if not isinstance(document_id, str) or not document_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed sparse candidate at rank %d: reason='missing or empty document_id'",
                    rank,
                )
                continue

            if not isinstance(cand_project_id, str) or not cand_project_id.strip():
                discarded_count += 1
                logger.warning(
                    "Discarded malformed sparse candidate at rank %d: reason='missing or empty project_id'",
                    rank,
                )
                continue

            # Defensive project isolation check
            if clean_project_id is not None and cand_project_id.strip() != clean_project_id:
                discarded_count += 1
                logger.warning(
                    "Discarded leaked sparse candidate at rank %d: reason='project_id mismatch' "
                    "(expected='%s', actual='%s')",
                    rank,
                    clean_project_id,
                    cand_project_id,
                )
                continue

            cid = chunk_id.strip()
            # If candidate already appeared in sparse list, skip to preserve best/first rank
            if cid in seen_sparse_chunks:
                continue
            seen_sparse_chunks.add(cid)

            # Calculate RRF score contribution using 1-based rank
            score_contrib = 1.0 / (self._rrf_k + rank)
            doc_version = (
                candidate.document_version_id.strip()
                if (
                    getattr(candidate, "document_version_id", None)
                    and candidate.document_version_id.strip()
                )
                else None
            )

            sparse_score_val = (
                float(raw_score)
                if (isinstance(raw_score, (int, float)) and math.isfinite(raw_score))
                else None
            )

            sparse_metadata = dict(getattr(candidate, "metadata", None) or {})

            if cid in accumulator:
                # Deduplicate and merge: candidate appeared in both branches
                accumulator[cid]["score"] += score_contrib
                accumulator[cid]["sparse_rank"] = rank
                accumulator[cid]["sparse_score"] = sparse_score_val
                if accumulator[cid]["document_version_id"] is None and doc_version is not None:
                    accumulator[cid]["document_version_id"] = doc_version
                # Merge sparse metadata fields if not already populated from dense branch
                for k, v in sparse_metadata.items():
                    if k not in accumulator[cid]["metadata"]:
                        accumulator[cid]["metadata"][k] = v
            else:
                # Candidate appeared only in sparse search
                accumulator[cid] = {
                    "chunk_id": cid,
                    "document_id": document_id.strip(),
                    "project_id": cand_project_id.strip(),
                    "document_version_id": doc_version,
                    "score": score_contrib,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": sparse_score_val,
                    "metadata": sparse_metadata,
                }

        if discarded_count > 0:
            logger.warning("Fusion discarded %d malformed or leaked candidate entries", discarded_count)

        # 5. Deterministic Ordering: O(C log C)
        # Order by descending fused score; tie-break deterministically by ascending chunk_id
        sorted_entries = sorted(
            accumulator.values(),
            key=lambda item: (-item["score"], item["chunk_id"]),
        )

        # 6. Build Immutable FusedCandidate Models with 1-based Rank
        fused_candidates: list[FusedCandidate] = []
        for unified_rank, entry in enumerate(sorted_entries, start=1):
            fused_candidates.append(
                FusedCandidate(
                    chunk_id=entry["chunk_id"],
                    document_id=entry["document_id"],
                    project_id=entry["project_id"],
                    score=entry["score"],
                    rank=unified_rank,
                    dense_rank=entry["dense_rank"],
                    sparse_rank=entry["sparse_rank"],
                    dense_score=entry["dense_score"],
                    sparse_score=entry["sparse_score"],
                    document_version_id=entry["document_version_id"],
                    metadata=entry.get("metadata", {}),
                )
            )

        # 7. Apply Output Top-K Bound
        if effective_top_k is not None:
            fused_candidates = fused_candidates[:effective_top_k]

        return fused_candidates
