"""Deterministic output validator for query transformation responses."""

from dataclasses import dataclass
import re

from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of transformation output validation."""

    is_valid: bool
    cleaned_query: str
    reason: str


class TransformationOutputValidator:
    """Practical, deterministic validator for LLM query rewrite outputs.

    Verifies that the transformed query is non-empty, stays within length limits,
    does not contain an explanatory answer instead of a retrieval query, and preserves
    technical identifiers, error codes, and requirement IDs from the original query.
    """

    # Obvious conversational answer prefixes that indicate the model answered the query
    ANSWER_PREFIXES: tuple[str, ...] = (
        "yes,",
        "yes.",
        "no,",
        "no.",
        "the answer is",
        "based on",
        "according to",
        "here is what",
        "here's what",
        "here is the answer",
        "in summary",
        "sure,",
        "certainly,",
        "i found that",
        "as an ai",
        "to answer your question",
        "there is no",
    )

    # Patterns identifying technical identifiers, codes, and version numbers
    IDENTIFIER_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\b"),  # BRD-102, ERR-404, API-v2, voyage-4
        re.compile(r"(?:\bC\+\+(?!\w)|\bC#(?!\w)|\b\.NET\b)", re.IGNORECASE),  # C++, C#, .NET
        re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE),                  # v2.1, v1
    )


    def __init__(self, max_query_length: int = 2000, preserve_identifiers: bool = True) -> None:
        self._max_query_length = max_query_length
        self._preserve_identifiers = preserve_identifiers

    def clean_output(self, raw_output: str) -> str:
        """Strip markdown code blocks, prefixes, and quotation wrappers."""
        text = raw_output.strip()

        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                # Remove first and last line
                text = "\n".join(lines[1:-1]).strip()

        # Remove common preamble labels
        for label in ("rewritten query:", "search query:", "query:", "transformed query:"):
            if text.lower().startswith(label):
                text = text[len(label):].strip()

        # Remove outer quotes if wrapped
        if len(text) >= 2 and (
            (text.startswith('"') and text.endswith('"'))
            or (text.startswith("'") and text.endswith("'"))
        ):
            text = text[1:-1].strip()

        # Collapse whitespace
        return " ".join(text.split())

    def extract_identifiers(self, text: str) -> set[str]:
        """Extract technical identifiers, requirement IDs, error codes, and versions."""
        identifiers: set[str] = set()
        for pattern in self.IDENTIFIER_PATTERNS:
            for match in pattern.findall(text):
                identifiers.add(match.strip())
        return identifiers

    def validate(self, raw_output: str, original_query: str) -> ValidationResult:
        """Validate transformed query output deterministically."""
        cleaned = self.clean_output(raw_output)

        # 1. Non-empty check
        if not cleaned:
            logger.warning("Query transformation rejected: output is empty.")
            return ValidationResult(is_valid=False, cleaned_query="", reason="empty_output")

        # 2. Maximum length constraint
        if len(cleaned) > self._max_query_length:
            logger.warning(
                "Query transformation rejected: length (%d) exceeds max (%d).",
                len(cleaned),
                self._max_query_length,
            )
            return ValidationResult(
                is_valid=False,
                cleaned_query=cleaned,
                reason="length_exceeded",
            )

        # 3. Practical answer detection (obvious answer prefixes)
        lower_cleaned = cleaned.lower()
        for prefix in self.ANSWER_PREFIXES:
            if lower_cleaned.startswith(prefix):
                logger.warning(
                    "Query transformation rejected: answer prefix detected ('%s').", prefix
                )
                return ValidationResult(
                    is_valid=False,
                    cleaned_query=cleaned,
                    reason="answer_detected",
                )

        # 4. Identifier preservation check
        if self._preserve_identifiers:
            orig_identifiers = self.extract_identifiers(original_query)
            if orig_identifiers:
                lower_cleaned_for_id = lower_cleaned
                for identifier in orig_identifiers:
                    if identifier.lower() not in lower_cleaned_for_id:
                        logger.warning(
                            "Query transformation rejected: required identifier '%s' missing from transformed query.",
                            identifier,
                        )
                        return ValidationResult(
                            is_valid=False,
                            cleaned_query=cleaned,
                            reason="identifier_dropped",
                        )

        return ValidationResult(is_valid=True, cleaned_query=cleaned, reason="valid")
