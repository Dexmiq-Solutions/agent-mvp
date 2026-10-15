"""Intent Router Agent package exports."""

from agents.recommendation.router.agent import IntentRouterAgent
from agents.recommendation.router.prompts import (
    ROUTER_SYSTEM_INSTRUCTION,
    build_router_user_prompt,
)

__all__ = [
    "IntentRouterAgent",
    "ROUTER_SYSTEM_INSTRUCTION",
    "build_router_user_prompt",
]
