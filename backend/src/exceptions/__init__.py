"""Application domain exceptions."""

from exceptions.acquisition import (
    AcquisitionConfigurationError,
    AcquisitionError,
    InvalidSourceReferenceError,
    SourceDiscoveryError,
    UnsupportedDocumentError,
)
from exceptions.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseSessionError,
)
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseValidationError,
)
from exceptions.ingestion import (
    DocumentRetrievalError,
    IngestionConfigurationError,
    IngestionError,
    InvalidIngestionInputError,
    UnsupportedDocumentTypeError,
)
from exceptions.cleaning import (
    CleaningConfigurationError,
    CleaningError,
    CleaningProcessingError,
    InvalidCleaningInputError,
)
from exceptions.normalization import (
    InvalidNormalizationInputError,
    NormalizationConfigurationError,
    NormalizationError,
    NormalizationProcessingError,
)
from exceptions.chunking import (
    ChunkingConfigurationError,
    ChunkingError,
    ChunkingProcessingError,
    InvalidChunkingInputError,
)
from exceptions.metadata_enrichment import (
    InvalidMetadataEnrichmentInputError,
    MetadataEnrichmentConfigurationError,
    MetadataEnrichmentError,
    MetadataEnrichmentProcessingError,
)
from exceptions.contextual_enrichment import (
    ContextualEnrichmentConfigurationError,
    ContextualEnrichmentError,
    ContextualEnrichmentProcessingError,
    ContextualEnrichmentProviderError,
    InvalidContextualEnrichmentInputError,
)
from exceptions.indexing import (
    IndexingConfigurationError,
    IndexingConnectionError,
    IndexingError,
    InvalidIndexingInputError,
    IndexingOperationError,
    IndexingPartialFailureError,
)
from exceptions.parsing import (
    DocumentExtractionError,
    InvalidParsingInputError,
    ParsingError,
    UnsupportedDocumentTypeError as ParsingUnsupportedDocumentTypeError,
)
from exceptions.storage import (
    BucketNotFoundError,
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageBucketError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageUploadError,
)
from exceptions.vector import (
    CollectionConfigurationError,
    CollectionNotFoundError,
    VectorDeletionError,
    VectorInputValidationError,
    VectorSearchError,
    VectorStoreAuthenticationError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    QueryEmbeddingError,
    QueryLengthExceededError,
    QueryPreprocessingError,
    QueryTransformationError,
    RetrievalError,
    TransformationFallbackLimitExceededError,
    TransformationProviderError,
    TransformationTimeoutError,
    TransformationUnavailableError,
    TransformationValidationError,
    VectorRetrievalError,
    KeywordRetrievalError,
    SparseEncodingError,
)

__all__ = [
    # Acquisition Exceptions
    "AcquisitionError",
    "AcquisitionConfigurationError",
    "SourceDiscoveryError",
    "UnsupportedDocumentError",
    "InvalidSourceReferenceError",
    # Ingestion Exceptions
    "IngestionError",
    "IngestionConfigurationError",
    "InvalidIngestionInputError",
    "UnsupportedDocumentTypeError",
    "DocumentRetrievalError",
    # Storage Exceptions
    "StorageError",
    "StorageConfigurationError",
    "StorageConnectionError",
    "StorageAuthenticationError",
    "BucketNotFoundError",
    "ObjectNotFoundError",
    "StorageUploadError",
    "StorageDownloadError",
    "StorageDeleteError",
    "StorageBucketError",
    # Database Exceptions
    "DatabaseError",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseSessionError",
    # Embedding Exceptions
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
    "EmbeddingInputValidationError",
    "EmbeddingRequestError",
    "EmbeddingResponseValidationError",
    # Vector Store Exceptions
    "VectorStoreError",
    "VectorStoreConfigurationError",
    "VectorStoreConnectionError",
    "VectorStoreAuthenticationError",
    "CollectionNotFoundError",
    "CollectionConfigurationError",
    "VectorUpsertError",
    "VectorSearchError",
    "VectorDeletionError",
    "VectorInputValidationError",
    # Parsing Exceptions
    "ParsingError",
    "ParsingUnsupportedDocumentTypeError",
    "DocumentExtractionError",
    "InvalidParsingInputError",
    # Cleaning Exceptions
    "CleaningError",
    "InvalidCleaningInputError",
    "CleaningProcessingError",
    "CleaningConfigurationError",
    # Normalization Exceptions
    "NormalizationError",
    "InvalidNormalizationInputError",
    "NormalizationProcessingError",
    "NormalizationConfigurationError",
    # Chunking Exceptions
    "ChunkingError",
    "InvalidChunkingInputError",
    "ChunkingProcessingError",
    "ChunkingConfigurationError",
    # Metadata Enrichment Exceptions
    "MetadataEnrichmentError",
    "InvalidMetadataEnrichmentInputError",
    "MetadataEnrichmentProcessingError",
    "MetadataEnrichmentConfigurationError",
    # Contextual Enrichment Exceptions
    "ContextualEnrichmentError",
    "InvalidContextualEnrichmentInputError",
    "ContextualEnrichmentConfigurationError",
    "ContextualEnrichmentProcessingError",
    "ContextualEnrichmentProviderError",
    # Indexing Exceptions
    "IndexingError",
    "InvalidIndexingInputError",
    "IndexingConfigurationError",
    "IndexingConnectionError",
    "IndexingOperationError",
    "IndexingPartialFailureError",
    # Retrieval Exceptions
    "RetrievalError",
    "QueryPreprocessingError",
    "InvalidQueryError",
    "EmptyQueryError",
    "QueryLengthExceededError",
    "QueryTransformationError",
    "TransformationValidationError",
    "TransformationProviderError",
    "TransformationTimeoutError",
    "TransformationUnavailableError",
    "TransformationFallbackLimitExceededError",
    "QueryEmbeddingError",
    "VectorRetrievalError",
    "KeywordRetrievalError",
    "SparseEncodingError",
]

