"""Unit tests for Intent Router Agent (Step 1C).

Covers:
- Routing to each of the 4 Profiles (explain, recommend, conflict, communication).
- Mixed request rule D24 (explain + recommend -> recommend).
- High-stakes triage and specific reasons.
- Missing facts identification for Communication profile.
- Ambiguous reference causing low confidence.
- Bounded retry and fallback when model fails.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.recommendation.contracts import (
    AgentRequest,
    RequestContext,
    ResolvedReference,
    RouteDecision,
    UserContext,
)
from agents.recommendation.router.agent import IntentRouterAgent


class MockRouterChatModel(BaseChatModel):
    """Deterministic mock model for Intent Router tests."""

    def __init__(self, decision: RouteDecision | None = None, should_fail: bool = False):
        super().__init__()
        self._decision = decision
        self._should_fail = should_fail

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self._should_fail:
            raise RuntimeError("Simulated LLM network/provider failure")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mock"))])

    def with_structured_output(self, schema, **kwargs):
        mock_bound = MagicMock()
        if self._should_fail:
            mock_bound.invoke.side_effect = RuntimeError("Simulated structured output failure")
        else:
            mock_bound.invoke.return_value = self._decision
        return mock_bound

    @property
    def _llm_type(self) -> str:
        return "mock-router-model"


def make_context(message: str, refs: list[ResolvedReference] | None = None, facts: dict[str, str] | None = None) -> RequestContext:
    req = AgentRequest(
        thread_id="t-1",
        brd_id="b-1",
        message=message,
        user=UserContext(user_id="u-1", tenant_id="u-1"),
    )
    return RequestContext(
        request=req,
        resolved_references=refs or [],
        user_provided_facts=facts or {},
    )


def test_route_to_explain():
    decision = RouteDecision(
        profile="explain",
        high_stakes=False,
        high_stakes_reasons=[],
        missing_facts=[],
        confidence="high",
        short_reason="User asks to explain an existing requirement.",
    )
    model = MockRouterChatModel(decision=decision)
    agent = IntentRouterAgent(model=model)

    ctx = make_context("Explain point 4 of section 6.")
    result = agent.route_sync(ctx)

    assert result.profile == "explain"
    assert result.confidence == "high"
    assert not result.high_stakes


def test_route_mixed_request_to_recommend():
    # D24: explain + alternatives -> recommend
    decision = RouteDecision(
        profile="recommend",
        high_stakes=False,
        high_stakes_reasons=[],
        missing_facts=[],
        confidence="high",
        short_reason="Mixed request: explain + alternatives routes to recommend.",
    )
    model = MockRouterChatModel(decision=decision)
    agent = IntentRouterAgent(model=model)

    ctx = make_context("Explain section 6 and suggest alternatives.")
    result = agent.route_sync(ctx)

    assert result.profile == "recommend"
    assert result.confidence == "high"


def test_route_to_conflict_high_stakes():
    decision = RouteDecision(
        profile="conflict",
        high_stakes=True,
        high_stakes_reasons=["compliance_or_legal", "hard_to_undo"],
        missing_facts=[],
        confidence="high",
        short_reason="Contradiction between data deletion and audit retention.",
    )
    model = MockRouterChatModel(decision=decision)
    agent = IntentRouterAgent(model=model)

    ctx = make_context("Section 2.3 contradicts section 3.2. Which one is right?")
    result = agent.route_sync(ctx)

    assert result.profile == "conflict"
    assert result.high_stakes is True
    assert "compliance_or_legal" in result.high_stakes_reasons


def test_route_to_communication_with_missing_facts():
    decision = RouteDecision(
        profile="communication",
        high_stakes=True,
        high_stakes_reasons=["client_facing", "large_cost_or_schedule"],
        missing_facts=["cause", "impact", "new_date", "mitigation"],
        confidence="high",
        short_reason="Client delay communication request without timeline facts.",
    )
    model = MockRouterChatModel(decision=decision)
    agent = IntentRouterAgent(model=model)

    ctx = make_context("We are going to miss the deadline. How do we tell the client?")
    result = agent.route_sync(ctx)

    assert result.profile == "communication"
    assert result.high_stakes is True
    assert "client_facing" in result.high_stakes_reasons
    assert len(result.missing_facts) == 4
    assert "new_date" in result.missing_facts


def test_ambiguous_reference_lowers_confidence():
    decision = RouteDecision(
        profile="explain",
        high_stakes=False,
        high_stakes_reasons=[],
        missing_facts=[],
        confidence="high",  # Model thought high, but reference is ambiguous
        short_reason="Explain section",
    )
    model = MockRouterChatModel(decision=decision)
    agent = IntentRouterAgent(model=model)

    ambiguous_ref = ResolvedReference(
        raw_text="that section",
        section_id=None,
        confidence="ambiguous",
        candidates=["2.3", "3.2"],
    )
    ctx = make_context("Explain that section.", refs=[ambiguous_ref])
    result = agent.route_sync(ctx)

    # Post-processing must have downgraded confidence to low
    assert result.confidence == "low"


def test_bounded_retry_and_fallback():
    # Model fails consistently
    model = MockRouterChatModel(should_fail=True)
    agent = IntentRouterAgent(model=model)

    ctx = make_context("A completely confusing message")
    result = agent.route_sync(ctx)

    # Fallback decision
    assert result.profile == "recommend"
    assert result.confidence == "low"
    assert "fallback triggered" in result.short_reason.lower()
