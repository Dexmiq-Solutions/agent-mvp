# End-to-End Indexing Pipeline

## 1. Overview & Objective

The **End-to-End Indexing Pipeline** connects the modular RAG indexing components into one unified, production-grade workflow. It takes a verified `DocumentVersion` from the **Document Processing Lifecycle**, executes structured document extraction, structural chunking, hybrid metadata enrichment, dual representation generation (dense + sparse), and atomically persists searchable project knowledge into both **PostgreSQL** (Content Store) and **Qdrant** (Vector Index Store).

### Core Architectural Invariants
- **DocumentVersion as Unit of Indexing**: The pipeline operates strictly on physical document versions identified by `document_version_id`, maintaining tenant and document identity throughout.
- **Strict Tenant & Project Isolation**: Every stage validates that `project_id` matches across source files, chunks, relational rows, and vector payloads.
- **Short PostgreSQL Transactions**: Transactions are isolated strictly to chunk persistence operations and never span external provider network calls (such as Voyage AI API or Qdrant cluster requests).
- **Dual Storage Separation**:
  - **PostgreSQL (`chunks`)**: The authoritative Content Store for verbatim chunk content, hierarchy coordinates, contextual headers, and metadata used for retrieval hydration and prompt context assembly.
  - **Qdrant (`test_chunks` / production collection)**: The Vector Index Store holding dense embeddings (`voyage-4`), sparse representations (`technical_hash`), and point payloads for hybrid search and filtering.
- **Idempotency & Version Supercedence**: Re-indexing a document version replaces prior records cleanly. Indexing a new version V2 automatically prunes superseded V1 vector points from Qdrant while preserving historical versions in PostgreSQL.

---

## 2. End-to-End Architecture & Sequence

```
DocumentVersion (status: PENDING)
       │
       ▼ [DocumentProcessingService] validates ownership & transitions to INDEXING
BaseDocumentProcessingPipeline / DefaultDocumentProcessingPipeline
       │
       ▼
EndToEndIndexingService
 ├── 1. Acquisition
 │      └─ Verify object exists in Object Storage (Supabase) via StorageAcquisitionService
 ├── 2. Ingestion
 │      └─ Stream raw bytes, determine MIME & canonical type via StorageIngestionService
 ├── 3. Parsing & Extraction
 │      └─ Extract hierarchical elements (headings, paragraphs, tables, code) via DocumentParsingService
 ├── 4. Cleaning
 │      └─ Strip control chars, clean excessive whitespace, preserve code blocks via DocumentCleaningService
 ├── 5. Normalization
 │      └─ NFKC canonical unicode & newline normalization via DocumentNormalizationService
 ├── 6. Structure-Aware Chunking
 │      └─ Split hierarchically respecting section boundaries & overlap via DocumentChunkingService
 ├── 7. Metadata Enrichment
 │      └─ Extract provenance, structural coordinates, and characteristics via DocumentMetadataEnrichmentService
 ├── 8. Optional Contextual Enrichment
 │      └─ Attach contextual summaries if enabled (bypassed if disabled) via DocumentContextualEnrichmentService
 ├── 9. Representation Generation
 │      ├─ Extract representation texts (verbatim or contextualized)
 │      ├─ Generate dense embeddings via Voyage AI (voyage-4, batched)
 │      └─ Generate sparse representations via BaseSparseEncoder (technical_hash)
 ├── 10. PostgreSQL Chunk Persistence
 │      └─ Atomic short transaction in ChunkPersistenceService:
 │         - Cleanly replaces any prior chunks for the document in `chunks` table
 │         - Inserts ChunkModel entities (chunk_id, project_id, content, contextual_content, metadata)
 ├── 11. Qdrant Vector Indexing
 │      └─ Batched upsert via DocumentIndexingService:
 │         - Deterministic point IDs: uuid5(NAMESPACE, f"{project_id}:{document_id}:{chunk_id}:{version_id}")
 │         - Named vectors: dense + sparse
 │         - Payload: project_id, document_id, version_id, chunk_id, chunk_index, section_path, heading
 └── 12. Stale Version Pruning
        └─ Query PostgreSQL for prior READY versions of document
           Call Qdrant delete_by_filter to remove obsolete vector points
       │
       ▼ [IndexingPipelineReport returned to Lifecycle Service]
DocumentVersion (status: READY, indexed_at: current_timestamp)
```

---

## 3. Stage Details & Contracts

### Stage 1: Acquisition (`rag.acquisition`)
- **Service**: `StorageAcquisitionService` (`BaseAcquisitionService`)
- **Action**: Validates that the file exists in the designated Supabase bucket (`documents`) under the project-prefixed path (`{project_id}/{document_id}/v{version_number}/{filename}`).
- **Failure**: Raises `SourceDiscoveryError` or `UnsupportedDocumentError` if missing or unrecognized format.

### Stage 2: Ingestion (`rag.ingestion`)
- **Service**: `StorageIngestionService` (`BaseIngestionService`)
- **Action**: Streams raw bytes from storage and constructs an `IngestedDocument` envelope preserving source metadata, original filename, and canonical MIME type.

### Stage 3: Parsing & Extraction (`rag.parsing`)
- **Service**: `DocumentParsingService`
- **Supported Formats**: Markdown (`.md`), Plaintext (`.txt`), Word (`.docx`), PDF (`.pdf`).
- **Action**: Parses the raw document into typed `ParsedElement` items (Headings, Paragraphs, Code blocks, Tables), preserving document order, heading levels, and hierarchical parentage.

### Stage 4: Cleaning (`rag.cleaning`)
- **Service**: `DocumentCleaningService`
- **Action**: Sanitizes text elements, strips unprintable control characters and invalid non-spacing sequences, while carefully preserving indentation and syntax inside code blocks and table structures.

### Stage 5: Normalization (`rag.normalization`)
- **Service**: `DocumentNormalizationService`
- **Action**: Applies NFKC canonical Unicode normalization and standardizes line breaks (`\n`) without semantic drift or summarization.

### Stage 6: Structure-Aware Chunking (`rag.chunking`)
- **Service**: `DocumentChunkingService`
- **Contract**: Generates `DocumentChunk` entities with stable `chunk_id` format `{document_id}_chunk_{chunk_index}`. Respects maximum and minimum chunk size constraints, heading boundaries, and recursive splitting for oversized sections.

### Stage 7: Hybrid Metadata Enrichment (`rag.metadata_enrichment`)
- **Service**: `DocumentMetadataEnrichmentService`
- **Output**: `EnrichedDocument` holding `EnrichedChunk` records with structured `ChunkMetadata` covering provenance, structural section hierarchy, character counts, word counts, and language detection.

### Stage 8: Optional Contextual Enrichment (`rag.contextual_enrichment`)
- **Service**: `DocumentContextualEnrichmentService`
- **Behavior**:
  - Controlled by configuration setting `CONTEXTUAL_ENRICHMENT_ENABLED` (defaults to `False`).
  - When disabled: Passes through unchanged without invoking external LLMs or adding latency.
  - When enabled: Generates document-level situational context prepended to chunk content for representation generation, stored in `contextual_content`.

### Stage 9: Representation Generation (`rag.embeddings` & `rag.indexing.representations`)
- **Dense Embeddings**: Generated via `BaseEmbeddingProvider` (`VoyageEmbeddingProvider` using model `voyage-4`, dimension `1024` or configured).
- **Sparse Representations**: Generated via `generate_sparse_representations` using `technical_hash` strategy (producing index/value pairs for lexical/BM25-style match).
- **Input Text**: Extracted via `extract_representation_texts(contextual_doc)` (uses `contextual_content` if enriched, falling back to original `content`).

### Stage 10: PostgreSQL Chunk Persistence (`rag.indexing.persistence`)
- **Service**: `ChunkPersistenceService`
- **Table**: `chunks` (`ChunkModel`)
- **Columns Persisted**:
  - `chunk_id`: Primary key (`{document_id}_chunk_{index}`).
  - `project_id`: Project foreign key (enforcing tenant isolation).
  - `document_id`: Document foreign key.
  - `document_version_id`: Version foreign key.
  - `content`: Verbatim chunk text.
  - `contextual_content`: Contextualized text if enriched.
  - `chunk_index`: 0-based ordering index.
  - `heading`, `heading_level`, `section_path`, `element_types`.
  - `chunk_metadata`: JSON column containing serialized `ChunkMetadata`.
- **Transaction Boundary**: Wrapped in an isolated short async transaction. Prior chunks for `project_id` and `document_id` are deleted before insertion, guaranteeing idempotency.

### Stage 11: Qdrant Vector Indexing (`rag.indexing.service`)
- **Service**: `DocumentIndexingService`
- **Point Identity**: Stable UUIDv5 point ID generated via `generate_point_id(project_id, document_id, chunk_id, document_version_id)`.
- **Vectors**: Dual named vectors:
  - Dense: `[float, ...]` matching embedding model dimension.
  - Sparse: `SparseVector(indices=[...], values=[...])`.
- **Payload**: Contains relational foreign keys (`project_id`, `document_id`, `document_version_id`, `chunk_id`, `chunk_index`), section path, and filterable metadata.
- **Batching & Retries**: Batched upserts with exponential backoff and transient error recovery.

### Stage 12: Stale Version Pruning (`rag.indexing.orchestrator`)
- When indexing version `V2` succeeds:
  1. Queries PostgreSQL for existing `READY` versions of `document_id` where `id != V2.id`.
  2. For each superseded version, invokes `delete_document_version(project_id, document_id, old_version_id)` in Qdrant.
  3. Ensures vector search only returns chunks from the current version.

---

## 4. Dual Storage Architecture

| Dimension | PostgreSQL Content Store (`chunks`) | Qdrant Vector Store (`test_chunks`) |
| :--- | :--- | :--- |
| **Role** | Authoritative content, hydration, context assembly | Approximate Nearest Neighbor (ANN) search & filtering |
| **Primary Key** | `chunk_id` (`{document_id}_chunk_{index}`) | Deterministic RFC 4122 UUIDv5 point ID |
| **Search Capabilities** | Primary key lookup, relational foreign keys | Dense semantic search, Sparse lexical search, hybrid fusion |
| **Payload/Data** | Full verbatim text, contextual content, metadata JSON | Embedding vectors, sparse vectors, filter payload |
| **Transaction Lifetime** | Short atomic transaction (<50ms) | Network API calls with bounded retries |

---

## 5. Idempotency & Tenant Isolation Guarantees

1. **Tenant Isolation**:
   - Every service checks `project_id`.
   - `ChunkPersistenceService` validates that every chunk's `project_id` matches the document's `project_id`. Mismatches raise an immediate `InvalidIndexingInputError`.
   - Qdrant queries and upserts mandate `project_id` payload filtering.

2. **Idempotency**:
   - Re-running indexing for the same version replaces PostgreSQL chunks and updates Qdrant points idempotently without generating duplicates or primary key conflicts.

3. **Fault Tolerance**:
   - If any stage fails (e.g. storage download, parsing, embedding, vector store upsert), the error is intercepted by `DocumentProcessingService`.
   - The version transitions to `FAILED` with sanitized error diagnostics (`error_message`), masking any API keys or connection strings.
   - External provider failures do not leave uncommitted database locks.

---

## 6. Verification & Test Suite

The indexing pipeline is covered by comprehensive unit and integration tests:

| Test File | Scope |
| :--- | :--- |
| `tests/test_indexing_pipeline.py` | Full 10-stage integration, chunk persistence, lifecycle execution, error masking, stale pruning, batching |
| `tests/test_indexing.py` | `DocumentIndexingService` point generation, dual vector indexing, retries, Qdrant integration |
| `tests/test_document_processing_lifecycle.py` | State machine transitions (`PENDING` -> `INDEXING` -> `READY`/`FAILED`), duplicate prevention |
| `tests/test_chunk_hydration.py` | Hydration of PostgreSQL chunks from Qdrant candidate results |
| `tests/test_chunking.py` | Structure-aware chunking contracts, heading hierarchy, recursive splitting |
