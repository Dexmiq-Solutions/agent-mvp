"""BRD Rewriter Sub-Agent package."""

from agents.brd.rewriter.agent import (
    BRDRewriterAgent,
    BRDRewriterContext,
    BRDRewriterResult,
    DocumentEdit,
    get_rewriter_system_instruction_path,
    load_rewriter_system_instruction,
)

__all__ = [
    "BRDRewriterAgent",
    "BRDRewriterContext",
    "BRDRewriterResult",
    "DocumentEdit",
    "get_rewriter_system_instruction_path",
    "load_rewriter_system_instruction",
]
