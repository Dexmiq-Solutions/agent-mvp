"""Deterministic normalization rules and element-specific strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from normalization.models import NormalizationRuleType
from parsing.models import ElementType, ParsedElement

# Pre-compiled regex patterns for deterministic normalization
SAFE_CONTROL_CHARS_REGEX = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff\u200e\u200f\u202a-\u202e]"
)
MULTI_SPACE_REGEX = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_REGEX = re.compile(r"\n(?:[ \t]*\n){2,}")
LIST_MARKER_REGEX = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
TABLE_DELIMITER_CELL_REGEX = re.compile(r"^:?-+:?$")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration options for structure-aware deterministic normalization."""

    unicode_form: str = "NFC"
    normalize_whitespace: bool = True
    normalize_line_endings: bool = True
    normalize_blank_lines: bool = True
    max_consecutive_blank_lines: int = 1
    strip_control_characters: bool = True
    preserve_code_indentation: bool = True


# ==============================================================================
# Low-level Deterministic Transform Functions
# ==============================================================================


def normalize_line_endings(text: str) -> tuple[str, bool]:
    """Convert all CRLF (\\r\\n) and CR (\\r) line endings to standard LF (\\n).

    Args:
        text: Input string.

    Returns:
        Tuple of (normalized_text, was_changed).
    """
    if "\r" not in text:
        return text, False
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, normalized != text


def normalize_unicode_nfc(text: str, form: str = "NFC") -> tuple[str, bool]:
    """Apply Unicode canonical composition (NFC) to ensure consistent character representation.

    Does not transliterate, translate, or strip meaningful characters.

    Args:
        text: Input string.
        form: Unicode normalization form ('NFC', 'NFKC', etc. Default: 'NFC').

    Returns:
        Tuple of (normalized_text, was_changed).
    """
    normalized = unicodedata.normalize(form, text)
    return normalized, normalized != text


def strip_safe_control_characters(text: str) -> tuple[str, bool]:
    """Remove non-printable control characters and zero-width artifacts.

    Preserves standard formatting whitespace: horizontal tab (\\t) and line feed (\\n).

    Args:
        text: Input string.

    Returns:
        Tuple of (cleaned_text, was_changed).
    """
    cleaned = SAFE_CONTROL_CHARS_REGEX.sub("", text)
    return cleaned, cleaned != text


def collapse_whitespace_in_line(line: str) -> str:
    """Collapse multiple consecutive spaces or tabs into a single space within a line."""
    return MULTI_SPACE_REGEX.sub(" ", line)


def normalize_blank_lines(text: str, max_consecutive: int = 1) -> tuple[str, bool]:
    """Normalize excessive consecutive blank lines inside multi-line text elements.

    Args:
        text: Input multi-line string.
        max_consecutive: Maximum allowed consecutive blank lines (Default: 1, meaning at most \\n\\n).

    Returns:
        Tuple of (normalized_text, was_changed).
    """
    # max_consecutive=1 means max 2 newlines (1 blank line). 3+ newlines collapsed to 2.
    replacement = "\n" * (max_consecutive + 1)
    # Pattern matching (max_consecutive + 1) or more empty lines
    pattern = re.compile(r"\n(?:[ \t]*\n){" + str(max_consecutive + 1) + r",}")
    normalized = pattern.sub(replacement, text)
    return normalized, normalized != text


# ==============================================================================
# Element-Specific Normalizer Strategies
# ==============================================================================


class BaseElementNormalizer(ABC):
    """Abstract interface for element-specific structure-aware normalization."""

    @abstractmethod
    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        """Normalize element content according to structural type requirements.

        Args:
            content: Raw element content.
            config: Normalization configuration settings.

        Returns:
            Tuple of (normalized_content, list_of_applied_rule_names).
        """


class HeadingNormalizer(BaseElementNormalizer):
    """Normalizer for heading elements (ElementType.HEADING).

    Enforces single-line representation, collapses internal horizontal whitespace,
    and strips leading/trailing spaces while preserving heading text and levels.
    """

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        # 1. Line endings
        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        # 2. Unicode normalization
        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        # 3. Safe control characters
        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        # 4. Whitespace: Headings must be single-line; replace any internal newlines/tabs with space
        original_before_ws = result
        if "\n" in result or "\t" in result:
            result = result.replace("\n", " ").replace("\t", " ")

        if config.normalize_whitespace:
            result = collapse_whitespace_in_line(result)
            result = result.strip()

        if result != original_before_ws:
            applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        return result, applied_rules


class ParagraphNormalizer(BaseElementNormalizer):
    """Normalizer for paragraphs and general text blocks (ElementType.PARAGRAPH, TEXT_BLOCK).

    Normalizes whitespace line-by-line, collapses internal multi-spaces,
    and standardizes excessive blank lines while preserving paragraph meaning.
    """

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        # 1. Line endings
        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        # 2. Unicode normalization
        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        # 3. Safe control characters
        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        # 4. Line-by-line whitespace normalization
        original_before_ws = result
        if config.normalize_whitespace:
            lines = result.split("\n")
            norm_lines: list[str] = []
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    norm_lines.append("")
                else:
                    norm_lines.append(collapse_whitespace_in_line(stripped_line))
            result = "\n".join(norm_lines)

        if result != original_before_ws:
            applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        # 5. Blank-line normalization inside multi-line paragraphs
        if config.normalize_blank_lines:
            result, changed = normalize_blank_lines(result, config.max_consecutive_blank_lines)
            if changed:
                applied_rules.append(NormalizationRuleType.BLANK_LINES.value)

        # Overall trim leading/trailing newlines
        result = result.strip()

        return result, applied_rules


class CodeBlockNormalizer(BaseElementNormalizer):
    """Normalizer for code and preformatted elements (ElementType.CODE_BLOCK).

    STRICTLY preserves intentional indentation, tabs, and multi-space formatting.
    Only normalizes line endings, Unicode representation, and removes safe control chars.
    """

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        # 1. Line endings
        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        # 2. Unicode normalization
        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        # 3. Safe control characters
        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        # 4. Trailing whitespace stripping on code lines (preserves leading indentation!)
        if config.normalize_whitespace:
            original_code = result
            lines = result.split("\n")
            # rstrip() each line to eliminate trailing dead spaces while keeping exact leading indentation
            norm_lines = [line.rstrip() for line in lines]
            result = "\n".join(norm_lines)
            # Strip excessive leading/trailing blank lines from the block
            result = result.strip("\n")
            if result != original_code:
                applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        return result, applied_rules


class TableNormalizer(BaseElementNormalizer):
    """Normalizer for tabular data (ElementType.TABLE).

    Preserves row sequence, column pipe delimiters, and alignment while
    standardizing cell padding, line endings, and Unicode representation.
    """

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        # 1. Line endings
        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        # 2. Unicode normalization
        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        # 3. Safe control characters
        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        # 4. Table cell whitespace normalization
        if config.normalize_whitespace:
            original_table = result
            lines = result.split("\n")
            norm_rows: list[str] = []

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                if stripped_line.startswith("|") and stripped_line.endswith("|"):
                    # Standard pipe-delimited table row
                    raw_cells = stripped_line.split("|")
                    # raw_cells has empty string at index 0 and -1 due to outer pipes
                    inner_cells = raw_cells[1:-1]
                    norm_cells: list[str] = []
                    for cell in inner_cells:
                        clean_cell = cell.strip()
                        if TABLE_DELIMITER_CELL_REGEX.match(clean_cell):
                            # Delimiter cell (e.g. '---', ':---:', '---:')
                            norm_cells.append(f" {clean_cell} ")
                        else:
                            # Content cell: collapse multiple spaces
                            cell_text = collapse_whitespace_in_line(clean_cell)
                            norm_cells.append(f" {cell_text} ")
                    norm_rows.append(f"|{'|'.join(norm_cells)}|")
                else:
                    # Non-pipe row (e.g. tab or spaced columns)
                    norm_rows.append(collapse_whitespace_in_line(stripped_line))

            result = "\n".join(norm_rows)
            if result != original_table:
                applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        return result.strip(), applied_rules


class ListItemNormalizer(BaseElementNormalizer):
    """Normalizer for list items (ElementType.LIST_ITEM).

    Preserves list marker, numbering, and hierarchy indentation while
    standardizing body text whitespace, line endings, and Unicode representation.
    """

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        # 1. Line endings
        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        # 2. Unicode normalization
        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        # 3. Safe control characters
        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        # 4. Whitespace: preserve list marker and indentation, collapse body whitespace
        if config.normalize_whitespace:
            original_item = result
            match = LIST_MARKER_REGEX.match(result)
            if match:
                indent, marker, body = match.groups()
                clean_body = collapse_whitespace_in_line(body.strip())
                result = f"{indent}{marker} {clean_body}"
            else:
                result = collapse_whitespace_in_line(result.strip())

            if result != original_item:
                applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        return result, applied_rules


class DefaultNormalizer(BaseElementNormalizer):
    """Safe fallback normalizer for unrecognized or generic element types."""

    def normalize(
        self,
        content: str,
        config: NormalizationConfig,
    ) -> tuple[str, list[str]]:
        applied_rules: list[str] = []
        result = content

        if config.normalize_line_endings:
            result, changed = normalize_line_endings(result)
            if changed:
                applied_rules.append(NormalizationRuleType.LINE_ENDINGS.value)

        if config.unicode_form:
            result, changed = normalize_unicode_nfc(result, config.unicode_form)
            if changed:
                applied_rules.append(NormalizationRuleType.UNICODE_NFC.value)

        if config.strip_control_characters:
            result, changed = strip_safe_control_characters(result)
            if changed:
                applied_rules.append(NormalizationRuleType.CONTROL_CHARS.value)

        if config.normalize_whitespace:
            original = result
            lines = [collapse_whitespace_in_line(l.strip()) for l in result.split("\n")]
            result = "\n".join(lines).strip()
            if result != original:
                applied_rules.append(NormalizationRuleType.WHITESPACE.value)

        return result, applied_rules
