"""Configuration options for query preprocessing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPreprocessingConfig:
    """Configuration settings for deterministic query preprocessing.

    Attributes:
        max_query_length: Maximum allowed query character length before validation failure.
        unicode_form: Standard Unicode normalization form (default: "NFC").
    """

    max_query_length: int = 2000
    unicode_form: str = "NFC"
