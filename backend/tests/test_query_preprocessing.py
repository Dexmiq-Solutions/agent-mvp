"""Comprehensive unit and behavioral contract tests for Retrieval Query Preprocessing."""

import time
import unicodedata
import pytest

from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    QueryLengthExceededError,
    QueryPreprocessingError,
    RetrievalError,
)
from rag.retrieval import (
    EmptyQueryError as RagEmptyQueryError,
    InvalidQueryError as RagInvalidQueryError,
    ProcessedQuery as RagProcessedQuery,
    QueryLengthExceededError as RagQueryLengthExceededError,
    QueryPreprocessingConfig as RagQueryPreprocessingConfig,
    QueryPreprocessor as RagQueryPreprocessor,
    get_query_preprocessor as rag_get_query_preprocessor,
    preprocess_query as rag_preprocess_query,
)
from rag.retrieval import (
    ProcessedQuery,
    QueryPreprocessingConfig,
    QueryPreprocessingService,
    QueryPreprocessor,
    get_query_preprocessor,
    preprocess_query,
    reset_query_preprocessor,
)


@pytest.fixture(autouse=True)
def cleanup_preprocessor():
    """Reset singleton preprocessor before and after each test."""
    reset_query_preprocessor()
    yield
    reset_query_preprocessor()


# ==============================================================================
# 1. Validation Tests
# ==============================================================================


class TestQueryValidation:
    """Tests verifying strict query input validation."""

    def test_empty_string_rejected(self):
        """Empty string must raise EmptyQueryError."""
        preprocessor = QueryPreprocessor()
        with pytest.raises(EmptyQueryError) as exc_info:
            preprocessor.preprocess("")
        assert "cannot be empty or whitespace only" in str(exc_info.value)
        assert isinstance(exc_info.value, InvalidQueryError)
        assert isinstance(exc_info.value, QueryPreprocessingError)
        assert isinstance(exc_info.value, RetrievalError)

    @pytest.mark.parametrize(
        "whitespace_input",
        [
            " ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            "  \t  \n  \r  ",
        ],
    )
    def test_whitespace_only_rejected(self, whitespace_input: str):
        """Whitespace-only input must raise EmptyQueryError."""
        preprocessor = QueryPreprocessor()
        with pytest.raises(EmptyQueryError) as exc_info:
            preprocessor.preprocess(whitespace_input)
        assert "cannot be empty or whitespace only" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            None,
            123,
            45.67,
            [],
            ["query"],
            {"query": "hello"},
            object(),
        ],
    )
    def test_non_string_type_rejected(self, invalid_type):
        """Non-string inputs must raise InvalidQueryError."""
        preprocessor = QueryPreprocessor()
        with pytest.raises(InvalidQueryError) as exc_info:
            preprocessor.preprocess(invalid_type)
        assert "Query must be a string" in str(exc_info.value)

    def test_valid_short_query_succeeds(self):
        """Standard valid query must succeed."""
        preprocessor = QueryPreprocessor()
        result = preprocessor.preprocess("hello")
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == "hello"
        assert result.processed_query == "hello"
        assert not result.is_changed

    def test_exact_max_length_boundary_accepted(self):
        """Query exactly matching max_query_length must be accepted."""
        limit = 100
        config = QueryPreprocessingConfig(max_query_length=limit)
        preprocessor = QueryPreprocessor(config=config)

        exact_query = "a" * limit
        result = preprocessor.preprocess(exact_query)
        assert result.processed_query == exact_query
        assert len(result.processed_query) == limit

    def test_query_exceeding_max_length_rejected(self):
        """Query exceeding max_query_length by 1 character must raise QueryLengthExceededError."""
        limit = 100
        config = QueryPreprocessingConfig(max_query_length=limit)
        preprocessor = QueryPreprocessor(config=config)

        oversized_query = "a" * (limit + 1)
        with pytest.raises(QueryLengthExceededError) as exc_info:
            preprocessor.preprocess(oversized_query)

        assert exc_info.value.length == limit + 1
        assert exc_info.value.max_length == limit
        assert "exceeds maximum allowed limit" in str(exc_info.value)
        assert isinstance(exc_info.value, InvalidQueryError)


# ==============================================================================
# 2. Whitespace Normalization Tests
# ==============================================================================


class TestWhitespaceNormalization:
    """Tests verifying conservative whitespace handling."""

    def test_trim_leading_whitespace(self):
        """Leading whitespace must be trimmed."""
        result = preprocess_query("   hello world")
        assert result.original_query == "   hello world"
        assert result.processed_query == "hello world"
        assert result.is_changed

    def test_trim_trailing_whitespace(self):
        """Trailing whitespace must be trimmed."""
        result = preprocess_query("hello world   \n")
        assert result.original_query == "hello world   \n"
        assert result.processed_query == "hello world"
        assert result.is_changed

    def test_collapse_internal_repeated_spaces(self):
        """Multiple consecutive spaces must be collapsed to a single space."""
        raw = "What did the client say about   the launch date???"
        expected = "What did the client say about the launch date???"
        result = preprocess_query(raw)
        assert result.original_query == raw
        assert result.processed_query == expected
        assert result.is_changed

    def test_collapse_mixed_internal_whitespace(self):
        """Tabs, newlines, and carriage returns within the query must collapse to single space."""
        raw = "Query\twith\n\nmultiple\r\n\twhitespace   separators"
        expected = "Query with multiple whitespace separators"
        result = preprocess_query(raw)
        assert result.processed_query == expected


# ==============================================================================
# 3. Unicode Normalization Tests
# ==============================================================================


class TestUnicodeNormalization:
    """Tests verifying safe deterministic Unicode NFC normalization."""

    def test_unicode_combining_characters_canonical_composition(self):
        """Canonical decomposition (e + combining acute) must compose into precomposed é."""
        # Decomposed: 'e' + '\u0301' (COMBINING ACUTE ACCENT)
        decomposed = "cafe\u0301"
        # Precomposed: '\u00e9'
        precomposed = "caf\u00e9"

        assert decomposed != precomposed  # Different bit representations before NFC
        result = preprocess_query(decomposed)
        assert result.processed_query == precomposed
        assert unicodedata.is_normalized("NFC", result.processed_query)

    def test_unicode_precomposed_preserved(self):
        """Already NFC-composed Unicode characters must remain intact."""
        raw = "Résolution des problèmes d'accès au système naïvement résumé"
        result = preprocess_query(raw)
        assert result.processed_query == raw


# ==============================================================================
# 4. Strict Preservation Tests (Contract Protection)
# ==============================================================================


class TestPreservationContract:
    """Tests establishing the behavioral contract of what preprocessing must NOT do.

    These tests intentionally guard against future regressions where a developer
    might mistakenly add lowercasing, punctuation stripping, spell correction, or rewriting.
    """

    def test_preserves_technical_identifiers(self):
        """Technical identifiers, error codes, and version numbers must be preserved verbatim."""
        identifiers = [
            "ERR-404",
            "C++",
            "API-v2",
            "BRD-102",
            "ERR_401",
            "voyage-4",
            "Qdrant",
            "v1.2.3",
            "OAuth2.0",
        ]
        for ident in identifiers:
            raw = f"Issue with {ident} in production"
            result = preprocess_query(raw)
            assert ident in result.processed_query, f"Failed to preserve identifier '{ident}'"

    def test_preserves_casing(self):
        """Canonical processed query must strictly preserve character casing."""
        raw = "OpenAI API and Qdrant integration with BRD specifications"
        result = preprocess_query(raw)
        # Must NOT be lowercased
        assert result.processed_query == raw
        assert "OpenAI API" in result.processed_query
        assert "Qdrant" in result.processed_query
        assert "BRD" in result.processed_query

    def test_preserves_meaningful_punctuation(self):
        """Punctuation such as question marks, hyphens, underscores, plus signs must be preserved."""
        raw = "What is C++20 and ERR_404: why did API-v2 fail???"
        result = preprocess_query(raw)
        assert result.processed_query == raw
        assert "???" in result.processed_query
        assert "C++20" in result.processed_query
        assert "ERR_404:" in result.processed_query
        assert "API-v2" in result.processed_query

    def test_refuses_spell_correction(self):
        """Preprocessing must NOT correct spelling (spell correction belongs to query transformation)."""
        raw = "What did the clent say about the launch?"
        result = preprocess_query(raw)
        assert "clent" in result.processed_query
        assert "client" not in result.processed_query

    def test_refuses_semantic_rewriting(self):
        """Preprocessing must NOT rewrite queries semantically."""
        raw = "What did they say about launch?"
        result = preprocess_query(raw)
        assert result.processed_query == raw
        assert "What launch date did the client agree to?" != result.processed_query

    def test_refuses_query_expansion(self):
        """Preprocessing must NOT append synonyms or expansions."""
        raw = "launch date"
        result = preprocess_query(raw)
        assert result.processed_query == "launch date"
        for expansion in ["release", "go-live", "deployment"]:
            assert expansion not in result.processed_query

    def test_refuses_query_decomposition(self):
        """Preprocessing must NOT decompose complex queries into subqueries."""
        raw = "What is the launch date and who approved BRD-102?"
        result = preprocess_query(raw)
        assert result.processed_query == raw


# ==============================================================================
# 5. Output Structure and Determinism Tests
# ==============================================================================


class TestOutputStructureAndDeterminism:
    """Tests verifying ProcessedQuery model structure and deterministic execution."""

    def test_original_query_preserved_verbatim(self):
        """original_query must contain the exact raw query received."""
        raw = "   Leading and trailing with   spaces \t\n"
        result = preprocess_query(raw)
        assert result.original_query == raw
        assert result.processed_query == "Leading and trailing with spaces"
        assert result.is_changed

    def test_is_changed_property(self):
        """is_changed must accurately indicate whether normalization modified the text."""
        unchanged = preprocess_query("Exact query")
        assert not unchanged.is_changed

        changed = preprocess_query("  Exact query  ")
        assert changed.is_changed

    def test_to_dict_serialization(self):
        """to_dict() must return accurate serializable structure."""
        raw = "  Test query  "
        result = preprocess_query(raw)
        data = result.to_dict()
        assert data == {
            "original_query": raw,
            "processed_query": "Test query",
            "is_changed": True,
        }

    def test_model_repr_is_safe(self):
        """__repr__ must provide clean representation."""
        result = preprocess_query("Sample")
        assert "ProcessedQuery" in repr(result)
        assert "original_query='Sample'" in repr(result)
        assert "processed_query='Sample'" in repr(result)

    def test_deterministic_output_across_repeated_invocations(self):
        """Same input must produce identical output across repeated runs."""
        raw = "   Query with   irregular \t spaces  and C++ ERR-404  \n"
        preprocessor = QueryPreprocessor()
        first = preprocessor.preprocess(raw)

        for _ in range(50):
            subsequent = preprocessor.preprocess(raw)
            assert subsequent == first
            assert subsequent.original_query == first.original_query
            assert subsequent.processed_query == first.processed_query


# ==============================================================================
# 6. Service Pattern and Aliases
# ==============================================================================


class TestServicePatternAndAliases:
    """Tests verifying service singletons, re-exports, and aliases."""

    def test_query_preprocessing_service_alias(self):
        """QueryPreprocessingService must be an alias of QueryPreprocessor."""
        assert QueryPreprocessingService is QueryPreprocessor
        service = QueryPreprocessingService()
        result = service.preprocess("test query")
        assert result.processed_query == "test query"

    def test_singleton_lifecycle(self):
        """get_query_preprocessor must return same singleton until reset."""
        p1 = get_query_preprocessor()
        p2 = get_query_preprocessor()
        assert p1 is p2

        reset_query_preprocessor()
        p3 = get_query_preprocessor()
        assert p3 is not p1

    def test_rag_retrieval_re_exports(self):
        """backend/src/rag/retrieval must re-export all core components identically."""
        assert RagProcessedQuery is ProcessedQuery
        assert RagQueryPreprocessor is QueryPreprocessor
        assert RagQueryPreprocessingConfig is QueryPreprocessingConfig
        assert RagInvalidQueryError is InvalidQueryError
        assert RagEmptyQueryError is EmptyQueryError
        assert RagQueryLengthExceededError is QueryLengthExceededError

        result = rag_preprocess_query("   rag re-export test   ")
        assert result.processed_query == "rag re-export test"

        singleton = rag_get_query_preprocessor()
        assert isinstance(singleton, RagQueryPreprocessor)


# ==============================================================================
# 7. Performance and Latency
# ==============================================================================


class TestPerformance:
    """Tests verifying sub-millisecond, low-allocation execution."""

    def test_sub_millisecond_latency(self):
        """Single query preprocessing must complete in well under 1 millisecond."""
        preprocessor = QueryPreprocessor()
        query = (
            "What did the engineering team specify about Qdrant collection partitioning "
            "and BRD-102 indexing performance with voyage-4 embeddings in C++ API-v2?"
        )

        # Warm-up run
        preprocessor.preprocess(query)

        # Measure 100 iterations
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            preprocessor.preprocess(query)
        total_ms = (time.perf_counter() - start) * 1000.0
        avg_ms_per_query = total_ms / iterations

        # Query preprocessing is pure string manipulation and should average < 0.1ms
        assert avg_ms_per_query < 0.5, f"Query preprocessing too slow: {avg_ms_per_query:.4f}ms/query"
