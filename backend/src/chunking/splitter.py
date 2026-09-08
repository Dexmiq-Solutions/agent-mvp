"""Structure-aware section tree building and recursive splitting algorithms."""

import re
from typing import Any

from chunking.models import ChunkingConfig, DocumentChunk
from chunking.sizer import BaseChunkSizer
from parsing.models import ElementType, ParsedElement

# Pre-compiled regex patterns for progressive text splitting
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_REGEX = re.compile(r"(?<=[;,:])\s+")
WORD_SPLIT_REGEX = re.compile(r"\s+")


class SectionNode:
    """Represents a structural node in the document hierarchy tree.

    Organizes headings, child sections, and direct content elements in document order.
    """

    def __init__(
        self,
        heading_element: ParsedElement | None = None,
        heading_level: int = 0,
        heading_text: str | None = None,
        section_path: tuple[str, ...] = (),
    ) -> None:
        self.heading_element = heading_element
        self.heading_level = heading_level
        self.heading_text = heading_text
        self.section_path = section_path
        self.elements: list[ParsedElement] = []
        self.subsections: list["SectionNode"] = []

    def __repr__(self) -> str:
        return (
            f"SectionNode(heading={self.heading_text!r}, level={self.heading_level}, "
            f"elements={len(self.elements)}, subsections={len(self.subsections)})"
        )


def build_section_tree(elements: list[ParsedElement]) -> SectionNode:
    """Construct a nested section tree from a sequence of document elements in O(N) time.

    Uses a stack-based algorithm that accurately reflects heading nesting (H1 -> H2 -> H3)
    and attaches non-heading elements to their immediate enclosing section. If a document
    has no headings, all elements attach directly to the root node.

    Args:
        elements: Ordered list of ParsedElements from the document.

    Returns:
        Root SectionNode representing the top-level document structure.
    """
    root = SectionNode(
        heading_element=None,
        heading_level=0,
        heading_text=None,
        section_path=(),
    )
    stack: list[SectionNode] = [root]

    for elem in elements:
        if elem.element_type == ElementType.HEADING and elem.heading_level is not None:
            level = elem.heading_level
            # Pop stack until finding a parent section with lower heading level
            while len(stack) > 1 and stack[-1].heading_level >= level:
                stack.pop()

            path = elem.section_path or (elem.content.strip(),)
            node = SectionNode(
                heading_element=elem,
                heading_level=level,
                heading_text=elem.content.strip(),
                section_path=path,
            )
            stack[-1].subsections.append(node)
            stack.append(node)
        else:
            # Content elements belong to the currently active innermost section
            stack[-1].elements.append(elem)

    return root


class RecursiveTextSplitter:
    """Splits oversized textual content progressively across natural boundaries.

    Hierarchy:
        Paragraphs (\\n\\n)
            ↓
        Lines (\\n)
            ↓
        Sentences (punctuation boundary)
            ↓
        Clauses (;,:)
            ↓
        Words (spaces)
            ↓
        Character slices (guaranteed termination fallback)
    """

    def __init__(
        self,
        sizer: BaseChunkSizer,
        max_chunk_size: int,
        min_chunk_size: int = 0,
        chunk_overlap: int = 0,
    ) -> None:
        self._sizer = sizer
        self._max_size = max_chunk_size
        self._min_size = min_chunk_size
        self._overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Recursively split text until all slices satisfy max_chunk_size.

        Args:
            text: Input string to split.

        Returns:
            List of non-empty string slices within max_chunk_size.
        """
        stripped = text.strip()
        if not stripped:
            return []

        # Base case: text already satisfies size limit
        if self._sizer.fits(stripped, self._max_size):
            return [stripped]

        # Execute progressive recursive splitting
        raw_chunks = self._recursive_split(stripped, level=0)

        # Apply soft min_chunk_size preference: merge tiny tail fragments where safe
        return self._apply_soft_min_size(raw_chunks)

    def _recursive_split(self, text: str, level: int) -> list[str]:
        """Progressively partition text across boundary levels."""
        if self._sizer.fits(text, self._max_size):
            return [text]

        # Progressive boundary levels
        if level == 0:
            # Paragraph boundary
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            separator = "\n\n"
        elif level == 1:
            # Line boundary
            parts = [line.strip() for line in text.split("\n") if line.strip()]
            separator = "\n"
        elif level == 2:
            # Sentence boundary
            parts = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
            separator = " "
        elif level == 3:
            # Clause boundary
            parts = [c.strip() for c in CLAUSE_SPLIT_REGEX.split(text) if c.strip()]
            separator = " "
        elif level == 4:
            # Word boundary
            parts = [w for w in WORD_SPLIT_REGEX.split(text) if w]
            separator = " "
        else:
            # Hard character window fallback (guaranteed termination)
            return self._split_characters(text)

        # If splitting at this level didn't break text into multiple parts, advance to next level
        if len(parts) <= 1:
            return self._recursive_split(text, level + 1)

        # Pack fragments up to max_chunk_size
        chunks: list[str] = []
        current_acc: list[str] = []
        current_size = 0

        for part in parts:
            part_size = self._sizer.measure(part)

            # If an individual fragment exceeds max size, split it with the next finer level
            if part_size > self._max_size:
                if current_acc:
                    merged = separator.join(current_acc)
                    chunks.append(merged)
                    current_acc = []
                    current_size = 0

                sub_chunks = self._recursive_split(part, level + 1)
                chunks.extend(sub_chunks)
                continue

            # Check if adding part exceeds max_size
            added_size = part_size if not current_acc else self._sizer.measure(separator) + part_size
            if current_size + added_size <= self._max_size:
                current_acc.append(part)
                current_size += added_size
            else:
                if current_acc:
                    merged = separator.join(current_acc)
                    chunks.append(merged)

                # Minimal overlap: carry over trailing context if configured
                current_acc = self._compute_overlap_accumulator(current_acc, part, separator)
                if current_acc:
                    current_size = self._sizer.measure(separator.join(current_acc))
                else:
                    current_acc = [part]
                    current_size = part_size

        if current_acc:
            chunks.append(separator.join(current_acc))

        return chunks

    def _compute_overlap_accumulator(
        self,
        prev_acc: list[str],
        next_part: str,
        separator: str,
    ) -> list[str]:
        """Compute initial accumulator for next chunk with minimal configured overlap."""
        if self._overlap <= 0 or not prev_acc:
            return [next_part]

        # Take trailing items from prev_acc as overlap context up to self._overlap
        overlap_items: list[str] = []
        overlap_size = 0
        for item in reversed(prev_acc):
            item_size = self._sizer.measure(item)
            if overlap_size + item_size <= self._overlap:
                overlap_items.insert(0, item)
                overlap_size += item_size
            else:
                break

        if overlap_items:
            candidate = overlap_items + [next_part]
            candidate_size = self._sizer.measure(separator.join(candidate))
            if candidate_size <= self._max_size:
                return candidate

        return [next_part]

    def _split_characters(self, text: str) -> list[str]:
        """Slices unsplittable text into hard character windows with guaranteed termination."""
        chunks: list[str] = []
        n = len(text)
        step = max(1, self._max_size - self._overlap)
        idx = 0
        while idx < n:
            end = min(idx + self._max_size, n)
            chunk_slice = text[idx:end].strip()
            if chunk_slice:
                chunks.append(chunk_slice)
            idx += step
        return chunks

    def _apply_soft_min_size(self, chunks: list[str]) -> list[str]:
        """Merge tiny residual fragments with the preceding chunk where safe.

        Does not force merging across distinct structural units.
        """
        if len(chunks) <= 1 or self._min_size <= 0:
            return chunks

        merged: list[str] = []
        for i, chunk in enumerate(chunks):
            if not merged:
                merged.append(chunk)
                continue

            chunk_size = self._sizer.measure(chunk)
            if chunk_size < self._min_size:
                # Attempt merging with previous chunk
                prev = merged[-1]
                combined = f"{prev} {chunk}"
                if self._sizer.fits(combined, self._max_size):
                    merged[-1] = combined
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)

        return merged


class StructureAwareChunker:
    """Traverses a section tree and generates structure-aware DocumentChunk items."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config
        self._sizer = config.sizer
        self._splitter = RecursiveTextSplitter(
            sizer=self._sizer,
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        self._recursively_split_count = 0
        self._structural_chunks_count = 0
        self._fallback_chunks_count = 0

    @property
    def recursively_split_count(self) -> int:
        return self._recursively_split_count

    @property
    def structural_chunks_count(self) -> int:
        return self._structural_chunks_count

    @property
    def fallback_chunks_count(self) -> int:
        return self._fallback_chunks_count

    def chunk_document(
        self,
        document_id: str,
        project_id: str,
        elements: list[ParsedElement],
        document_version_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Perform structure-aware chunking on normalized elements in document order.

        Args:
            document_id: Source document ID.
            project_id: Multi-tenant project boundary.
            elements: Normalized elements in reading order.
            document_version_id: Optional document version ID.

        Returns:
            Ordered list of DocumentChunk instances.
        """
        if not elements:
            return []

        tree = build_section_tree(elements)
        chunks: list[DocumentChunk] = []

        # Process the section tree starting from root
        self._process_node(
            node=tree,
            document_id=document_id,
            project_id=project_id,
            document_version_id=document_version_id,
            chunks=chunks,
        )

        return chunks

    def _process_node(
        self,
        node: SectionNode,
        document_id: str,
        project_id: str,
        document_version_id: str | None,
        chunks: list[DocumentChunk],
    ) -> None:
        """Process direct elements of node, then recursively process subsections."""
        # 1. Process direct elements in this section
        direct_elements: list[ParsedElement] = []
        if node.heading_element is not None:
            direct_elements.append(node.heading_element)
        direct_elements.extend(node.elements)

        if direct_elements:
            self._pack_and_emit_elements(
                elements=direct_elements,
                node=node,
                document_id=document_id,
                project_id=project_id,
                document_version_id=document_version_id,
                chunks=chunks,
            )

        # 2. Process subsections independently, preserving their structural boundaries
        for subsection in node.subsections:
            self._process_node(
                node=subsection,
                document_id=document_id,
                project_id=project_id,
                document_version_id=document_version_id,
                chunks=chunks,
            )

    def _pack_and_emit_elements(
        self,
        elements: list[ParsedElement],
        node: SectionNode,
        document_id: str,
        project_id: str,
        document_version_id: str | None,
        chunks: list[DocumentChunk],
    ) -> None:
        """Group consecutive elements within the section up to max_chunk_size,

        recursively splitting any individual element that exceeds max_chunk_size.
        """
        accumulator: list[ParsedElement] = []
        accumulated_size = 0

        def flush_accumulator() -> None:
            nonlocal accumulator, accumulated_size
            if not accumulator:
                return

            joined_content = "\n\n".join(e.content.strip() for e in accumulator if e.content.strip())
            if joined_content:
                chunk = self._create_chunk(
                    content=joined_content,
                    source_elements=accumulator,
                    node=node,
                    document_id=document_id,
                    project_id=project_id,
                    document_version_id=document_version_id,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)
                if not chunk.section_path and chunk.heading is None:
                    self._fallback_chunks_count += 1
                else:
                    self._structural_chunks_count += 1

            accumulator = []
            accumulated_size = 0

        for elem in elements:
            content = elem.content.strip()
            if not content:
                continue

            elem_size = self._sizer.measure(content)

            # Check if this individual element exceeds max_chunk_size
            if elem_size > self._config.max_chunk_size:
                # Flush existing accumulated content before splitting oversized element
                flush_accumulator()

                # Recursively split oversized element
                self._recursively_split_count += 1
                split_slices = self._splitter.split_text(content)

                for slice_text in split_slices:
                    chunk = self._create_chunk(
                        content=slice_text,
                        source_elements=[elem],
                        node=node,
                        document_id=document_id,
                        project_id=project_id,
                        document_version_id=document_version_id,
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk)
                    if not chunk.section_path and chunk.heading is None:
                        self._fallback_chunks_count += 1
                    else:
                        self._structural_chunks_count += 1
                continue

            # Calculate size if added to accumulator
            separator_size = self._sizer.measure("\n\n") if accumulator else 0
            new_size = accumulated_size + separator_size + elem_size

            if new_size <= self._config.max_chunk_size:
                accumulator.append(elem)
                accumulated_size = new_size
            else:
                flush_accumulator()
                accumulator.append(elem)
                accumulated_size = elem_size

        flush_accumulator()

    def _create_chunk(
        self,
        content: str,
        source_elements: list[ParsedElement],
        node: SectionNode,
        document_id: str,
        project_id: str,
        document_version_id: str | None,
        chunk_index: int,
    ) -> DocumentChunk:
        """Construct a DocumentChunk with complete provenance and hierarchy information."""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        source_element_ids = tuple(e.element_id for e in source_elements)
        element_types = tuple(e.element_type for e in source_elements)

        # Consolidate metadata from source elements without duplicates
        consolidated_meta: dict[str, Any] = {}
        for elem in source_elements:
            for k, v in elem.metadata.items():
                if k not in consolidated_meta:
                    consolidated_meta[k] = v
        consolidated_meta["character_count"] = len(content)

        # Hierarchy & parent context
        section_path = node.section_path or (source_elements[0].section_path if source_elements else ())
        heading = node.heading_text
        heading_level = node.heading_level if node.heading_level > 0 else None
        parent_element_id = (
            node.heading_element.element_id if node.heading_element else None
        ) or (source_elements[0].parent_id if source_elements else None)

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            project_id=project_id,
            content=content,
            index=chunk_index,
            document_version_id=document_version_id,
            section_path=section_path,
            heading=heading,
            heading_level=heading_level,
            parent_element_id=parent_element_id,
            parent_chunk_id=None,
            source_element_ids=source_element_ids,
            element_types=element_types,
            metadata=consolidated_meta,
        )
