"""Comprehensive unit and behavioral contract tests for Retrieval Query Transformation."""

import asyncio
from typing import Any, Optional
import pytest

from exceptions.retrieval import (
    QueryTransformationError,
    RetrievalError,
    TransformationFallbackLimitExceededError,
    TransformationProviderError,
    TransformationTimeoutError,
    TransformationUnavailableError,
    TransformationValidationError,
)
from rag.retrieval import (
    AdaptiveTransformationPolicy as RagAdaptiveTransformationPolicy,
    BaseLLMClient as RagBaseLLMClient,
    PolicyDecision as RagPolicyDecision,
    QueryTransformationConfig as RagQueryTransformationConfig,
    QueryTransformationService as RagQueryTransformationService,
    RetrievalQuerySet as RagRetrievalQuerySet,
    get_query_transformation_service as rag_get_query_transformation_service,
    reset_query_transformation_service as rag_reset_query_transformation_service,
    transform_query as rag_transform_query,
)
from retrieval import (
    AdaptiveTransformationPolicy,
    BaseLLMClient,
    LLMQueryRewriteStrategy,
    OpenAICompatibleLLMClient,
    PolicyDecision,
    ProcessedQuery,
    QueryTransformationConfig,
    QueryTransformationService,
    RetrievalQuerySet,
    TransformationOutputValidator,
    get_query_transformation_service,
    preprocess_query,
    reset_query_transformation_service,
    transform_query,
)


class MockLLMClient(BaseLLMClient):
    """Predictable mock LLM client for testing without external API calls."""

    def __init__(
        self,
        canned_response: str = "Mock transformed query",
        should_fail_with: Optional[Exception] = None,
        fail_attempts: int = 0,
    ) -> None:
        self.canned_response = canned_response
        self.should_fail_with = should_fail_with
        self.fail_attempts = fail_attempts
        self.call_count = 0
        self.recorded_prompts: list[str] = []
        self.recorded_system_prompts: list[Optional[str]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        self.call_count += 1
        self.recorded_prompts.append(prompt)
        self.recorded_system_prompts.append(system_prompt)

        if self.should_fail_with:
            if self.fail_attempts == 0 or self.call_count <= self.fail_attempts:
                raise self.should_fail_with

        return self.canned_response


@pytest.fixture(autouse=True)
def cleanup_transformation_service():
    """Reset singleton service before and after each test."""
    reset_query_transformation_service()
    rag_reset_query_transformation_service()
    yield
    reset_query_transformation_service()
    rag_reset_query_transformation_service()


# ==============================================================================
# 1. Basic & Self-Contained Query Tests
# ==============================================================================


class TestBasicSelfContainedQueries:
    """Tests verifying behavior on standard self-contained user queries."""

    @pytest.mark.asyncio
    async def test_self_contained_query_bypasses_transformation(self):
        """Self-contained queries pass through with no LLM calls."""
        mock_client = MockLLMClient(canned_response="Something rewritten")
        service = QueryTransformationService(
            llm_client=mock_client,
        )

        query = "What is the timeout for API-v2?"
        result = await service.transform(query)

        assert isinstance(result, RetrievalQuerySet)
        assert result.original_query == "What is the timeout for API-v2?"
        assert result.transformed_query is None
        assert result.is_transformed is False
        assert result.queries == ("What is the timeout for API-v2?",)
        assert result.metadata.get("reason") == "self_contained"
        # Zero LLM calls made
        assert mock_client.call_count == 0

    @pytest.mark.asyncio
    async def test_accepts_both_string_and_processed_query(self):
        """Service seamlessly accepts raw strings or ProcessedQuery instances."""
        mock_client = MockLLMClient()
        service = QueryTransformationService(llm_client=mock_client)

        raw_result = await service.transform("How to configure Qdrant collection?")
        assert raw_result.original_query == "How to configure Qdrant collection?"

        processed = preprocess_query("How to configure Qdrant collection?")
        obj_result = await service.transform(processed)
        assert obj_result.original_query == "How to configure Qdrant collection?"
        assert obj_result.queries == ("How to configure Qdrant collection?",)

    @pytest.mark.asyncio
    async def test_whitespace_normalization_preserved(self):
        """Input with uncollapsed whitespace is normalized cleanly through preprocessing."""
        service = QueryTransformationService()
        result = await service.transform("   BRD-102    user    authentication   requirements   ")

        assert result.original_query == "BRD-102 user authentication requirements"
        assert result.queries == ("BRD-102 user authentication requirements",)


# ==============================================================================
# 2. Transformation Policy Tests
# ==============================================================================


class TestTransformationPolicy:
    """Tests verifying the deterministic, adaptive policy signals."""

    @pytest.fixture
    def policy(self):
        return AdaptiveTransformationPolicy()

    @pytest.mark.parametrize(
        "query_text,expected_reason",
        [
            ("what about that?", "conversational_continuation"),
            ("and the timeout?", "conversational_continuation"),
            ("what did we decide about this?", "conversational_continuation"),
            ("how does that affect UAT?", "conversational_continuation"),
            ("what happens if they reject it?", "conversational_continuation"),
            ("also what is the port?", "conversational_continuation"),
            ("how about the database credentials?", "conversational_continuation"),
            ("what if the server crashes?", "conversational_continuation"),
            ("so what is next?", "conversational_continuation"),
        ],
    )
    def test_conversational_continuations_trigger_transformation(self, policy, query_text, expected_reason):
        """Context-dependent conversational starters trigger transformation."""
        processed = preprocess_query(query_text)
        decision = policy.evaluate(processed)
        assert decision.should_transform is True
        assert decision.reason in ("conversational_continuation", "context_dependent_reference")

    @pytest.mark.parametrize(
        "deictic_query",
        [
            "why did it fail?",
            "is that required?",
            "how do these work?",
            "can we remove those?",
        ],
    )
    def test_deictic_pronouns_trigger_transformation(self, policy, deictic_query):
        """Short queries with unbound deictic pronouns trigger transformation."""
        processed = preprocess_query(deictic_query)
        decision = policy.evaluate(processed, conversation_context=["Previous discussion"])
        assert decision.should_transform is True
        assert decision.reason in ("conversational_continuation", "context_dependent_reference")


    @pytest.mark.parametrize(
        "self_contained_query",
        [
            "What is the timeout for API-v2?",
            "How to configure Qdrant collection?",
            "BRD-102 user authentication requirements",
            "PostgreSQL connection pooling configuration parameters",
            "Explain semantic search vs hybrid search in detail",
        ],
    )
    def test_self_contained_queries_bypass_transformation(self, policy, self_contained_query):
        """Self-contained queries evaluate to should_transform=False."""
        processed = preprocess_query(self_contained_query)
        decision = policy.evaluate(processed)
        assert decision.should_transform is False
        assert decision.reason == "self_contained"

    def test_fallback_flag_triggers_transformation(self, policy):
        """Fallback flag unconditionally triggers transformation."""
        processed = preprocess_query("What is the timeout for API-v2?")
        decision = policy.evaluate(processed, is_fallback=True)
        assert decision.should_transform is True
        assert decision.reason == "retrieval_fallback"


# ==============================================================================
# 3. LLM Query Rewriting & Output Validation Tests
# ==============================================================================


class TestLLMQueryRewriting:
    """Tests verifying LLM query rewriting, context injection, and validation."""

    @pytest.mark.asyncio
    async def test_successful_contextual_rewrite(self):
        """Context-dependent query is rewritten into standalone retrieval query."""
        mock_client = MockLLMClient(
            canned_response="What happens when the client rejects UAT approval?"
        )
        service = QueryTransformationService(llm_client=mock_client)

        context = "We were discussing the UAT approval process."
        query = "What happens if they reject it?"

        result = await service.transform(query, conversation_context=context)

        assert result.is_transformed is True
        assert result.original_query == "What happens if they reject it?"
        assert result.transformed_query == "What happens when the client rejects UAT approval?"
        # Additive non-destructive guarantee: [original, transformed]
        assert result.queries == (
            "What happens if they reject it?",
            "What happens when the client rejects UAT approval?",
        )
        assert result.strategy_used == "llm_rewrite"
        assert mock_client.call_count == 1
        # Context was included in prompt
        assert "UAT approval process" in mock_client.recorded_prompts[0]

    @pytest.mark.asyncio
    async def test_prompt_includes_system_instructions(self):
        """System prompt contains strict instructions forbidding answering."""
        mock_client = MockLLMClient(canned_response="Rewritten query")
        service = QueryTransformationService(llm_client=mock_client)

        await service.transform("what about that?", conversation_context="Some topic")

        assert mock_client.recorded_system_prompts[0] is not None
        assert "DO NOT answer" in mock_client.recorded_system_prompts[0]
        assert "Preserve all technical identifiers" in mock_client.recorded_system_prompts[0]


# ==============================================================================
# 4. Identifier Preservation Tests
# ==============================================================================


class TestIdentifierPreservation:
    """Tests verifying technical codes and identifiers are never stripped."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "BRD-102",
            "ERR-404",
            "API-v2",
            "voyage-4",
            "C++",
            "v2.1",
        ],
    )
    def test_validator_extracts_identifiers(self, identifier: str):
        """Validator correctly extracts technical tokens."""
        validator = TransformationOutputValidator()
        extracted = validator.extract_identifiers(f"Check status for {identifier} in module")
        assert any(identifier.lower() == item.lower() for item in extracted)

    @pytest.mark.asyncio
    async def test_rewrite_dropping_identifier_is_rejected(self):
        """If rewrite drops a critical identifier, it fails validation and falls back to original."""
        # Original has API-v2, but rewrite drops it to generic 'API'
        mock_client = MockLLMClient(
            canned_response="What is the standard API timeout requirement?"
        )
        service = QueryTransformationService(llm_client=mock_client)

        # Force fallback to trigger rewrite
        result = await service.transform(
            "What is the timeout for API-v2?",
            is_fallback=True,
        )

        # Because API-v2 was dropped, validation fails and original query is safely retained
        assert result.is_transformed is False
        assert result.transformed_query is None
        assert result.queries == ("What is the timeout for API-v2?",)
        assert result.metadata.get("reason") == "transformation_failed"

    @pytest.mark.asyncio
    async def test_rewrite_preserving_identifier_is_accepted(self):
        """When rewrite preserves technical identifier, it is accepted."""
        mock_client = MockLLMClient(
            canned_response="What is the default client connection timeout for API-v2?"
        )
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "and the timeout for API-v2?",
            conversation_context="Discussing API-v2 gateway",
        )

        assert result.is_transformed is True
        assert "API-v2" in (result.transformed_query or "")
        assert result.queries[0] == "and the timeout for API-v2?"
        assert result.queries[1] == "What is the default client connection timeout for API-v2?"


# ==============================================================================
# 5. Output Validation & Answer Detection Tests
# ==============================================================================


class TestOutputValidation:
    """Tests verifying invalid, empty, or answer-like responses are safely caught."""

    @pytest.mark.parametrize(
        "answer_output",
        [
            "Yes, the timeout for API-v2 is configured at 30 seconds.",
            "No, you cannot change the database port in production.",
            "The answer is 500 milliseconds according to BRD-102.",
            "Based on the documentation, UAT approval requires two sign-offs.",
            "According to the specs, ERR-404 is returned on missing endpoints.",
            "Here is what happens when UAT is rejected.",
            "In summary, Qdrant stores document chunks.",
            "Sure, here is the answer you requested.",
            "Certainly, the system works as follows.",
        ],
    )
    @pytest.mark.asyncio
    async def test_answer_like_output_rejected_safely(self, answer_output: str):
        """Responses starting with answer phrases are rejected, falling back to original query."""
        mock_client = MockLLMClient(canned_response=answer_output)
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous topic",
        )

        assert result.is_transformed is False
        assert result.transformed_query is None
        assert result.queries == ("what about that?",)
        assert result.metadata.get("reason") == "transformation_failed"

    @pytest.mark.parametrize(
        "invalid_output",
        [
            "",
            "   ",
            '""',
            "```\n```",
            "rewritten query:",
        ],
    )
    @pytest.mark.asyncio
    async def test_empty_or_whitespace_output_rejected(self, invalid_output: str):
        """Empty or whitespace output falls back safely to original query."""
        mock_client = MockLLMClient(canned_response=invalid_output)
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous topic",
        )

        assert result.is_transformed is False
        assert result.queries == ("what about that?",)

    @pytest.mark.asyncio
    async def test_oversized_output_rejected(self):
        """Output exceeding max query length limit is rejected."""
        long_output = "search " * 500
        mock_client = MockLLMClient(canned_response=long_output)
        config = QueryTransformationConfig(max_query_length=50)
        service = QueryTransformationService(config=config, llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous topic",
        )

        assert result.is_transformed is False
        assert result.queries == ("what about that?",)


# ==============================================================================
# 6. Provider Failure & Resilience Tests
# ==============================================================================


class TestProviderFailureResilience:
    """Tests verifying provider timeouts and errors never crash retrieval."""

    @pytest.mark.asyncio
    async def test_provider_timeout_falls_back_to_original(self):
        """Provider timeout safely returns original query without raising."""
        mock_client = MockLLMClient(
            should_fail_with=TransformationTimeoutError("Request timed out")
        )
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous context",
        )

        assert result.is_transformed is False
        assert result.transformed_query is None
        assert result.queries == ("what about that?",)
        assert result.metadata.get("reason") == "transformation_failed"

    @pytest.mark.asyncio
    async def test_provider_unavailable_falls_back_to_original(self):
        """Provider connection failure safely returns original query."""
        mock_client = MockLLMClient(
            should_fail_with=TransformationUnavailableError("Connection refused")
        )
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous context",
        )

        assert result.is_transformed is False
        assert result.queries == ("what about that?",)

    @pytest.mark.asyncio
    async def test_provider_http_error_falls_back_to_original(self):
        """Provider HTTP 500 error safely returns original query."""
        mock_client = MockLLMClient(
            should_fail_with=TransformationProviderError("Internal Server Error")
        )
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Previous context",
        )

        assert result.is_transformed is False
        assert result.queries == ("what about that?",)


# ==============================================================================
# 7. Fallback Integration & Bounded Fallback Tests
# ==============================================================================


class TestFallbackIntegration:
    """Tests verifying bounded fallback behavior within the same component."""

    @pytest.mark.asyncio
    async def test_fallback_triggers_rewrite_for_otherwise_self_contained_query(self):
        """Fallback re-entry causes self-contained query to attempt rewrite."""
        mock_client = MockLLMClient(
            canned_response="API-v2 endpoint timeout specification and latency limits"
        )
        service = QueryTransformationService(llm_client=mock_client)

        # Attempt 1 without fallback: passes through directly
        pass1 = await service.transform("What is the timeout for API-v2?", is_fallback=False)
        assert pass1.is_transformed is False
        assert mock_client.call_count == 0

        # Attempt 2 with fallback: triggers rewrite
        pass2 = await service.transform(
            "What is the timeout for API-v2?",
            is_fallback=True,
            attempt=2,
        )
        assert pass2.is_transformed is True
        assert pass2.queries == (
            "What is the timeout for API-v2?",
            "API-v2 endpoint timeout specification and latency limits",
        )
        assert mock_client.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_attempts_are_strictly_bounded(self):
        """Attempts beyond max_fallback_attempts stop and retain original query."""
        mock_client = MockLLMClient(canned_response="Should never be called")
        # max_fallback_attempts = 1 allows attempt 1 (normal) and attempt 2 (fallback 1)
        config = QueryTransformationConfig(max_fallback_attempts=1)
        service = QueryTransformationService(config=config, llm_client=mock_client)

        # Attempt 3 exceeds max_fallback_attempts + 1
        result = await service.transform(
            "What is the timeout for API-v2?",
            is_fallback=True,
            attempt=3,
        )

        assert result.is_transformed is False
        assert result.metadata.get("reason") == "fallback_limit_reached"
        assert result.queries == ("What is the timeout for API-v2?",)
        assert mock_client.call_count == 0


# ==============================================================================
# 8. Configuration & Single Source of Truth Tests
# ==============================================================================


class TestConfiguration:
    """Tests verifying configuration loading and single source of truth."""

    def test_config_from_settings(self):
        """QueryTransformationConfig loads cleanly from application Settings."""
        config = QueryTransformationConfig.from_settings()
        assert config.enabled is True
        assert config.strategy == "llm_rewrite"
        assert config.model is not None
        assert config.max_fallback_attempts == 1

    @pytest.mark.asyncio
    async def test_disabled_transformation_bypasses_all_queries(self):
        """When enabled=False, all queries bypass transformation directly."""
        mock_client = MockLLMClient(canned_response="Transformed")
        config = QueryTransformationConfig(enabled=False)
        service = QueryTransformationService(config=config, llm_client=mock_client)

        result = await service.transform("what about that?", conversation_context="Context")
        assert result.is_transformed is False
        assert result.metadata.get("reason") == "disabled"
        assert mock_client.call_count == 0


# ==============================================================================
# 9. Project Isolation & Facade Re-export Tests
# ==============================================================================


class TestProjectIsolationAndFacade:
    """Tests verifying project isolation integrity and rag.retrieval facade."""

    @pytest.mark.asyncio
    async def test_project_id_preserved_in_metadata(self):
        """Project ID is preserved and not modified by transformation."""
        mock_client = MockLLMClient(canned_response="Rewritten query")
        service = QueryTransformationService(llm_client=mock_client)

        result = await service.transform(
            "what about that?",
            conversation_context="Context",
            project_id="proj-12345",
        )

        assert result.metadata.get("project_id") == "proj-12345"

    @pytest.mark.asyncio
    async def test_rag_retrieval_facade_matches_retrieval(self):
        """rag.retrieval exports match retrieval exports directly."""
        assert RagQueryTransformationService is QueryTransformationService
        assert RagRetrievalQuerySet is RetrievalQuerySet
        assert RagQueryTransformationConfig is QueryTransformationConfig
        assert RagAdaptiveTransformationPolicy is AdaptiveTransformationPolicy
        assert RagBaseLLMClient is BaseLLMClient
        assert RagPolicyDecision is PolicyDecision
        assert rag_get_query_transformation_service is get_query_transformation_service
        assert rag_transform_query is transform_query


# ==============================================================================
# 10. RetrievalQuerySet Domain Model Tests
# ==============================================================================


class TestRetrievalQuerySetContract:
    """Tests verifying the domain model contract of RetrievalQuerySet."""

    def test_untransformed_query_set(self):
        """Untransformed query set contains exactly the original query."""
        qs = RetrievalQuerySet(original_query="hello world")
        assert qs.original_query == "hello world"
        assert qs.transformed_query is None
        assert qs.is_transformed is False
        assert qs.queries == ("hello world",)
        assert qs.has_transformed_query is False
        serialized = qs.to_dict()
        assert serialized["queries"] == ["hello world"]

    def test_transformed_query_set(self):
        """Transformed query set contains both original and transformed queries additively."""
        qs = RetrievalQuerySet(
            original_query="hello world",
            transformed_query="greetings earth",
            is_transformed=True,
            strategy_used="llm_rewrite",
            metadata={"reason": "context_dependent"},
        )
        assert qs.queries == ("hello world", "greetings earth")
        assert qs.has_transformed_query is True
        assert qs.strategy_used == "llm_rewrite"
        assert qs.metadata["reason"] == "context_dependent"

    def test_identical_transformed_query_collapses_to_original(self):
        """If transformed query is identical to original, queries property returns only original."""
        qs = RetrievalQuerySet(
            original_query="hello world",
            transformed_query="hello world",
            is_transformed=True,
        )
        assert qs.queries == ("hello world",)
        assert qs.has_transformed_query is False
