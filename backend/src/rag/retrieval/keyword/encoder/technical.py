"""Deterministic technical sparse encoder for lexical retrieval."""

from collections import Counter
import re
from typing import Optional

import xxhash

from exceptions.retrieval import SparseEncodingError
from rag.retrieval.keyword.encoder.base import BaseSparseEncoder
from rag.retrieval.models import SparseVector


# Tokenizer regex preserving C++, compound identifiers, error codes, and technical symbols
# Examples: ERR-401, API-v2, BRD-102, user_id, C++, voyage-4, qdrant
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"c\+\+|[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)+|[a-zA-Z0-9_]+",
    re.IGNORECASE,
)
SUBTOKEN_SPLIT_PATTERN = re.compile(r"[-_.]")


ENGLISH_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
})


class TechnicalSparseEncoder(BaseSparseEncoder):
    """Deterministic lexical sparse encoder optimized for technical documentation.

    Algorithm:
        1. Technical tokenization: extracts technical symbols (e.g. C++), compound terms
           (e.g. ERR-401, API-v2, BRD-102, user_id), and standard alphanumeric tokens.
           Also extracts constituent sub-tokens for compound identifiers.
        2. Deterministic term hashing: maps normalized terms to 32-bit unsigned integer
           dimension indices using xxhash.xxh32_intdigest.
        3. Saturated term-frequency weighting: applies BM25-inspired term-frequency saturation
           formula w = (tf * (k1 + 1)) / (tf + k1) to prevent high-frequency term dominance.
        4. Sorted sparse vector creation: produces strictly sorted, unique indices and
           finite non-negative weights conforming to Qdrant's sparse vector contract.

    Performance:
        Performs local CPU computation with O(n) complexity relative to token count.
        Does not require external model inference, GPU, or network round trips.
    """

    def __init__(
        self,
        k1: float = 1.2,
        subtoken_weight_multiplier: float = 0.5,
        version: str = "1.0",
    ) -> None:
        """Initialize TechnicalSparseEncoder.

        Args:
            k1: Term frequency saturation parameter (default: 1.2).
            subtoken_weight_multiplier: Weight multiplier for constituent sub-tokens (default: 0.5).
            version: Version string for compatibility contracts (default: '1.0').
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be positive, got {k1}.")
        if subtoken_weight_multiplier < 0:
            raise ValueError(
                f"subtoken_weight_multiplier cannot be negative, got {subtoken_weight_multiplier}."
            )
        self._k1 = k1
        self._subtoken_weight_multiplier = subtoken_weight_multiplier
        self._version = version

    @property
    def strategy_name(self) -> str:
        """Return the strategy identifier."""
        return "technical_hash"

    @property
    def version(self) -> str:
        """Return the encoder version identifier."""
        return self._version

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into technical terms including preserved compound terms and sub-tokens.

        Args:
            text: Text to tokenize.

        Returns:
            list[str]: Sequence of extracted technical tokens and constituent sub-tokens.
        """
        if not isinstance(text, str):
            raise SparseEncodingError(f"Input text must be a string, got '{type(text).__name__}'.")
        clean_text = text.strip()
        if not clean_text:
            return []
        tokens = TECHNICAL_TOKEN_PATTERN.findall(clean_text.lower())
        all_tokens: list[str] = []
        for token in tokens:
            is_compound = "-" in token or "_" in token or "." in token or "+" in token
            if not is_compound and token in ENGLISH_STOP_WORDS:
                continue
            all_tokens.append(token)
            if is_compound:
                parts = SUBTOKEN_SPLIT_PATTERN.split(token)
                for part in parts:
                    if part and part != token and part not in ENGLISH_STOP_WORDS:
                        all_tokens.append(part)
        return all_tokens

    def _tokenize_and_weigh(self, text: str) -> SparseVector:
        """Extract terms, compute saturated weights, and generate sorted SparseVector."""
        if not isinstance(text, str):
            raise SparseEncodingError(
                f"Input text must be a string, got '{type(text).__name__}'."
            )

        clean_text = text.strip()
        if not clean_text:
            return SparseVector(indices=(), values=())

        tokens = TECHNICAL_TOKEN_PATTERN.findall(clean_text.lower())
        if not tokens:
            return SparseVector(indices=(), values=())

        # Count primary tokens
        raw_counts = Counter(tokens)
        weighted_counts: dict[str, float] = {}

        for token, count in raw_counts.items():
            is_compound = "-" in token or "_" in token or "." in token or "+" in token
            if not is_compound and token in ENGLISH_STOP_WORDS:
                continue
            weighted_counts[token] = weighted_counts.get(token, 0.0) + float(count)
            # If token is compound (e.g. err-401, api-v2, user_id), add sub-tokens
            if is_compound:
                parts = SUBTOKEN_SPLIT_PATTERN.split(token)
                for part in parts:
                    if part and part != token and part not in ENGLISH_STOP_WORDS:
                        weighted_counts[part] = (
                            weighted_counts.get(part, 0.0)
                            + float(count) * self._subtoken_weight_multiplier
                        )

        # Apply term-frequency saturation and map terms to 32-bit hash indices
        term_weights: dict[int, float] = {}
        for term, eff_tf in weighted_counts.items():
            val = (eff_tf * (self._k1 + 1.0)) / (eff_tf + self._k1)
            idx = xxhash.xxh32_intdigest(term.encode("utf-8"))
            rounded_val = round(float(val), 4)
            # In case of 32-bit hash collision, preserve maximum weight
            if idx in term_weights:
                term_weights[idx] = max(term_weights[idx], rounded_val)
            else:
                term_weights[idx] = rounded_val

        sorted_indices = tuple(sorted(term_weights.keys()))
        sorted_values = tuple(term_weights[idx] for idx in sorted_indices)

        return SparseVector(indices=sorted_indices, values=sorted_values)

    def encode_query(self, query: str) -> SparseVector:
        """Encode a retrieval query into a SparseVector.

        Args:
            query: Verbatim retrieval query text.

        Returns:
            SparseVector: Sorted non-negative dimension indices and positive weights.

        Raises:
            SparseEncodingError: If input is invalid.
        """
        try:
            return self._tokenize_and_weigh(query)
        except SparseEncodingError:
            raise
        except Exception as exc:
            raise SparseEncodingError(
                f"Failed to encode query into sparse representation: {exc}",
                original_error=exc,
            ) from exc

    def encode_queries(self, queries: list[str]) -> list[SparseVector]:
        """Encode a batch of retrieval queries into SparseVector representations.

        Args:
            queries: Sequence of retrieval query strings.

        Returns:
            list[SparseVector]: SparseVector representations corresponding 1-to-1 with inputs.

        Raises:
            SparseEncodingError: If input is invalid.
        """
        if not isinstance(queries, (list, tuple)):
            raise SparseEncodingError(
                f"queries must be a list or tuple of strings, got '{type(queries).__name__}'."
            )
        return [self.encode_query(q) for q in queries]

    def encode_document(self, text: str) -> SparseVector:
        """Encode a document chunk text into a SparseVector.

        Args:
            text: Document chunk textual content.

        Returns:
            SparseVector: Sorted non-negative dimension indices and positive weights.

        Raises:
            SparseEncodingError: If input is invalid.
        """
        try:
            return self._tokenize_and_weigh(text)
        except SparseEncodingError:
            raise
        except Exception as exc:
            raise SparseEncodingError(
                f"Failed to encode document chunk into sparse representation: {exc}",
                original_error=exc,
            ) from exc

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode a batch of document chunk texts into SparseVector representations.

        Args:
            texts: Sequence of document chunk textual content strings.

        Returns:
            list[SparseVector]: SparseVector representations corresponding 1-to-1 with inputs.

        Raises:
            SparseEncodingError: If input is invalid.
        """
        if not isinstance(texts, (list, tuple)):
            raise SparseEncodingError(
                f"texts must be a list or tuple of strings, got '{type(texts).__name__}'."
            )
        return [self.encode_document(t) for t in texts]
