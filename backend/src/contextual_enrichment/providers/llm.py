"""LLM-based context provider executing model-driven contextual enrichment."""

import asyncio
from typing import Any

from app.core.logging import get_logger
from contextual_enrichment.base import BaseContextProvider, DocumentContext
from exceptions.contextual_enrichment import ContextualEnrichmentProviderError
from llm.base import BaseLLMClient
from metadata_enrichment.models import EnrichedChunk

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a contextual assistant helping index technical documents for search and retrieval. "
    "Your task is to provide a short, factual 1-2 sentence context that situates a document chunk "
    "within its broader document. Do not summarize the chunk itself; explain its place, purpose, "
    "or topic in the overall document. Do not invent information. If the chunk is fully self-contained "
    "or no meaningful additional context is needed, respond with 'NONE'."
)


class LLMContextProvider(BaseContextProvider):
    """Context provider using an LLM to generate concise situational context for chunks."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_concurrency: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self._client = llm_client
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    @property
    def provider_name(self) -> str:
        """Return provider identifier including the active model name."""
        return f"llm:{self._client.model_name}"

    def _build_prompt(self, chunk: EnrichedChunk, context: DocumentContext) -> str:
        """Construct the prompt providing document and chunk context."""
        doc_info = []
        if context.original_filename:
            doc_info.append(f"Document: {context.original_filename}")
        if context.document_summary:
            doc_info.append(f"Document Summary: {context.document_summary}")
        elif context.headings_hierarchy:
            doc_info.append(f"Document Outline: {' > '.join(context.headings_hierarchy[:5])}")

        section_info = ""
        if chunk.enriched_metadata and chunk.enriched_metadata.heading_path_str:
            section_info = chunk.enriched_metadata.heading_path_str
        elif chunk.section_path:
            section_info = " > ".join(chunk.section_path)
        elif chunk.heading:
            section_info = chunk.heading

        prompt_lines = [
            f"<document_info>\n{chr(10).join(doc_info) if doc_info else 'General document'}\n</document_info>",
        ]
        if section_info:
            prompt_lines.append(f"<section_context>\n{section_info}\n</section_context>")
        prompt_lines.append(f"<chunk>\n{chunk.content}\n</chunk>")
        prompt_lines.append(
            "Please provide a concise 1-2 sentence contextual description to situate this chunk within the document. "
            "Respond with only the context, or 'NONE' if no additional context is needed."
        )
        return "\n\n".join(prompt_lines)

    async def generate_context(
        self,
        chunk: EnrichedChunk,
        context: DocumentContext,
    ) -> str | None:
        """Generate LLM contextual description for a single chunk."""
        prompt = self._build_prompt(chunk, context)
        async with self._semaphore:
            try:
                response = await self._client.complete(
                    prompt=prompt,
                    system_prompt=self._system_prompt,
                )
            except Exception as exc:
                logger.error(
                    "LLM contextual enrichment call failed for chunk '%s': %s",
                    chunk.chunk_id,
                    str(exc),
                )
                raise ContextualEnrichmentProviderError(
                    f"LLM provider error for chunk '{chunk.chunk_id}': {str(exc)}",
                    original_error=exc,
                ) from exc

        trimmed = response.strip() if response else ""
        if not trimmed or trimmed.upper() in {"NONE", "NONE.", "N/A", "NO CONTEXT NEEDED"}:
            return None

        return trimmed

    async def generate_context_batch(
        self,
        chunks: list[EnrichedChunk],
        context: DocumentContext,
    ) -> list[str | None]:
        """Generate LLM context for multiple chunks concurrently within semaphore bounds."""
        tasks = [self.generate_context(chunk, context) for chunk in chunks]
        return await asyncio.gather(*tasks)
