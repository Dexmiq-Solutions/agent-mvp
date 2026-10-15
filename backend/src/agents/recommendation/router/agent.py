"""Intent Router Agent implementation for Step 1C of Phase 1.

Responsible for:
- Selecting exactly one Profile from the starting set: explain, recommend, conflict, communication.
- Applying the mixed-request rule (D24).
- Determining high_stakes triage and recording specific reasons.
- Identifying missing required facts from context.
- Assessing confidence (flagging low confidence for ambiguous references or unclear requests).
- Returning a structured RouteDecision with bounded retries and graceful fallback.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.recommendation.config import RecommendationConfig, create_router_model
from agents.recommendation.contracts import RequestContext, RouteDecision
from agents.recommendation.router.prompts import (
    ROUTER_SYSTEM_INSTRUCTION,
    build_router_user_prompt,
)
from observability.logging import get_logger

logger = get_logger(__name__)


class IntentRouterAgent:
    """Lightweight Intent Router Agent for the Recommendation Workflow."""

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        config: Optional[RecommendationConfig] = None,
    ) -> None:
        """Initialize IntentRouterAgent.

        Args:
            model: Optional pre-configured BaseChatModel instance.
            config: Optional RecommendationConfig instance used if model is omitted.
        """
        self._config = config or RecommendationConfig.from_settings()
        self._model = model or create_router_model(self._config)

        # Structured output binding for RouteDecision
        try:
            self._structured_model = self._model.with_structured_output(RouteDecision)
        except Exception as exc:
            logger.warning(
                "Could not bind structured output directly on model: %s. Will parse output manually.",
                exc,
            )
            self._structured_model = None

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    async def route(self, context: RequestContext) -> RouteDecision:
        """Route user request asynchronously with bounded retries and fallback."""
        return self._execute_route(context)

    def route_sync(self, context: RequestContext) -> RouteDecision:
        """Route user request synchronously with bounded retries and fallback."""
        return self._execute_route(context)

    def _execute_route(self, context: RequestContext) -> RouteDecision:
        """Core execution logic with bounded retries and deterministic fallback."""
        user_prompt = build_router_user_prompt(
            message=context.request.message,
            resolved_references=[r.model_dump() for r in context.resolved_references],
            history_summary=context.history_summary,
            user_provided_facts=context.user_provided_facts,
        )

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_INSTRUCTION),
            HumanMessage(content=user_prompt),
        ]

        try:
            from observability.llm_telemetry import get_telemetry_callback_handler
            cb = get_telemetry_callback_handler()
            call_config = {"callbacks": [cb]} if cb else {}
        except Exception:
            call_config = {}

        # Bounded retry: attempt up to 2 times
        for attempt in range(1, 3):
            try:
                logger.info(
                    "Intent Router invoking model (attempt %d/2, model: %s)",
                    attempt,
                    self._config.router_model,
                )
                if self._structured_model is not None:
                    decision = self._structured_model.invoke(messages, config=call_config)
                    if isinstance(decision, RouteDecision):
                        logger.info(
                            "Intent Router received RouteDecision: profile='%s', high_stakes=%s, confidence='%s', rationale='%s'",
                            decision.profile,
                            decision.high_stakes,
                            decision.confidence,
                            decision.short_reason,
                        )
                        return self._post_process_decision(decision, context)
                    if isinstance(decision, dict):
                        parsed = RouteDecision.model_validate(decision)
                        logger.info(
                            "Intent Router validated RouteDecision from dict: profile='%s', high_stakes=%s, confidence='%s', rationale='%s'",
                            parsed.profile,
                            parsed.high_stakes,
                            parsed.confidence,
                            parsed.short_reason,
                        )
                        return self._post_process_decision(parsed, context)

                # Fallback to direct model invocation if structured binding not supported
                raw_response = self._model.invoke(messages, config=call_config)
                content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                parsed = self._extract_route_decision(content)
                if parsed is not None:
                    logger.info(
                        "Intent Router extracted RouteDecision from raw content: profile='%s', high_stakes=%s, confidence='%s', rationale='%s'",
                        parsed.profile,
                        parsed.high_stakes,
                        parsed.confidence,
                        parsed.short_reason,
                    )
                    return self._post_process_decision(parsed, context)

            except Exception as exc:
                logger.warning(
                    "Intent Router execution attempt %d failed: %s",
                    attempt,
                    exc,
                )

        # Fallback behavior when structured output fails
        logger.error(
            "Intent Router failed structured output after bounded retries. Triggering low-confidence fallback."
        )
        fallback_decision = RouteDecision(
            profile="recommend",
            high_stakes=False,
            high_stakes_reasons=[],
            missing_facts=[],
            confidence="low",
            short_reason="Router structured output parsing failed after bounded retries; fallback triggered.",
        )
        return self._post_process_decision(fallback_decision, context)

    def _post_process_decision(
        self,
        decision: RouteDecision,
        context: RequestContext,
    ) -> RouteDecision:
        """Enforce deterministic safety checks on the router decision."""
        # Check if references are ambiguous
        has_ambiguous_reference = any(
            r.confidence == "ambiguous" for r in context.resolved_references
        )
        if has_ambiguous_reference and decision.confidence == "high":
            # If a referenced entity is ambiguous, confidence must be low
            logger.info(
                "Lowering router confidence to 'low' due to ambiguous resolved reference."
            )
            decision.confidence = "low"
            if not decision.short_reason:
                decision.short_reason = "Ambiguous reference detected in request."
            else:
                decision.short_reason += " (Confidence lowered due to ambiguous reference)."

        return decision

    def _extract_route_decision(self, content: str) -> Optional[RouteDecision]:
        """Manually extract and parse JSON RouteDecision from text response."""
        try:
            # Check for JSON code block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                return RouteDecision.model_validate(data)
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                data = json.loads(content[start:end])
                return RouteDecision.model_validate(data)
        except Exception as exc:
            logger.debug("Failed manual JSON extraction: %s", exc)
        return None
