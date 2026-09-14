"""Structural analysis context for document cleaning."""

from collections import defaultdict
from typing import Any

from rag.parsing.models import ElementType, ParsedDocument, ParsedElement


def _normalize_text_key(text: str) -> str:
    """Normalize text key for frequency and comparison analysis.

    Removes excessive whitespace and lowercases without altering the actual content.
    """
    return " ".join(text.lower().split())


class DocumentStructureContext:
    """Pre-computed structural context and index for a ParsedDocument.

    Constructed in a single O(n) pass over document elements to avoid quadratic
    comparisons during deterministic and heuristic cleaning evaluations.
    """

    def __init__(self, document: ParsedDocument) -> None:
        self.document = document
        self.elements: list[ParsedElement] = document.elements
        self.total_elements: int = len(self.elements)

        self._element_by_id: dict[str, ParsedElement] = {}
        self._index_by_id: dict[str, int] = {}
        self._text_frequency: dict[str, int] = defaultdict(int)
        self._text_to_pages: dict[str, set[int]] = defaultdict(set)
        self._page_to_elements: dict[int, list[str]] = defaultdict(list)
        self._section_to_elements: dict[tuple[str, ...], list[str]] = defaultdict(list)

        self._has_page_info: bool = False

        self._build_index()

    def _build_index(self) -> None:
        """Populate structural indices in a single pass."""
        for idx, elem in enumerate(self.elements):
            self._element_by_id[elem.element_id] = elem
            self._index_by_id[elem.element_id] = idx

            # Track section associations
            if elem.section_path:
                self._section_to_elements[elem.section_path].append(elem.element_id)

            # Check for page metadata
            page_num: int | None = None
            for key in ("page_number", "page", "page_idx"):
                val = elem.metadata.get(key)
                if isinstance(val, int):
                    page_num = val
                    break
                elif isinstance(val, str) and val.isdigit():
                    page_num = int(val)
                    break

            if page_num is not None:
                self._has_page_info = True
                self._page_to_elements[page_num].append(elem.element_id)

            # Frequency tracking for short-to-medium textual elements (< 250 chars)
            stripped = elem.content.strip()
            if stripped and len(stripped) < 250:
                key = _normalize_text_key(stripped)
                self._text_frequency[key] += 1
                if page_num is not None:
                    self._text_to_pages[key].add(page_num)

    @property
    def has_page_info(self) -> bool:
        """Return whether the document contains page-level metadata."""
        return self._has_page_info

    @property
    def pages_count(self) -> int:
        """Return total count of distinct pages detected."""
        return len(self._page_to_elements)

    def get_element(self, element_id: str) -> ParsedElement | None:
        """Lookup an element by its ID."""
        return self._element_by_id.get(element_id)

    def get_index(self, element_id: str) -> int:
        """Return 0-based index of element in original sequence."""
        return self._index_by_id.get(element_id, -1)

    def get_previous_element(self, element_id: str) -> ParsedElement | None:
        """Return the immediately preceding element in document order, if any."""
        idx = self.get_index(element_id)
        if idx > 0:
            return self.elements[idx - 1]
        return None

    def get_next_element(self, element_id: str) -> ParsedElement | None:
        """Return the immediately following element in document order, if any."""
        idx = self.get_index(element_id)
        if 0 <= idx < self.total_elements - 1:
            return self.elements[idx + 1]
        return None

    def is_first_in_document(self, element_id: str) -> bool:
        """Check if element is the very first element in the document."""
        return self.get_index(element_id) == 0

    def is_last_in_document(self, element_id: str) -> bool:
        """Check if element is the very last element in the document."""
        return self.get_index(element_id) == self.total_elements - 1

    def is_first_in_page(self, element_id: str) -> bool:
        """Check if element is at the top of its page (if page info exists)."""
        elem = self.get_element(element_id)
        if not elem:
            return False
        for key in ("page_number", "page", "page_idx"):
            val = elem.metadata.get(key)
            if val is not None:
                try:
                    p = int(val)
                    page_elements = self._page_to_elements.get(p, [])
                    return bool(page_elements and page_elements[0] == element_id)
                except (ValueError, TypeError):
                    pass
        return False

    def is_last_in_page(self, element_id: str) -> bool:
        """Check if element is at the bottom of its page (if page info exists)."""
        elem = self.get_element(element_id)
        if not elem:
            return False
        for key in ("page_number", "page", "page_idx"):
            val = elem.metadata.get(key)
            if val is not None:
                try:
                    p = int(val)
                    page_elements = self._page_to_elements.get(p, [])
                    return bool(page_elements and page_elements[-1] == element_id)
                except (ValueError, TypeError):
                    pass
        return False

    def is_first_in_section(self, element_id: str) -> bool:
        """Check if element is the first non-heading child under its section."""
        elem = self.get_element(element_id)
        if not elem or not elem.section_path:
            return False
        sec_elements = self._section_to_elements.get(elem.section_path, [])
        # Find first non-heading element in this section
        for eid in sec_elements:
            e = self.get_element(eid)
            if e and e.element_type != ElementType.HEADING:
                return eid == element_id
        return False

    def is_last_in_section(self, element_id: str) -> bool:
        """Check if element is the last element in its section."""
        elem = self.get_element(element_id)
        if not elem or not elem.section_path:
            return False
        sec_elements = self._section_to_elements.get(elem.section_path, [])
        return bool(sec_elements and sec_elements[-1] == element_id)

    def get_repetition_count(self, content: str) -> int:
        """Return how many times this normalized text occurs in the document."""
        key = _normalize_text_key(content.strip())
        return self._text_frequency.get(key, 0)

    def get_page_distribution(self, content: str) -> set[int]:
        """Return the set of distinct page numbers where this content appears."""
        key = _normalize_text_key(content.strip())
        return self._text_to_pages.get(key, set())
