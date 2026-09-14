"""Unit tests for sparse vector models and TechnicalSparseEncoder."""

import pytest

from exceptions.retrieval import SparseEncodingError
from rag.retrieval.keyword.encoder import (
    BaseSparseEncoder,
    TechnicalSparseEncoder,
    get_sparse_encoder,
    reset_sparse_encoder,
)
from rag.retrieval.models import SparseVector


@pytest.fixture(autouse=True)
def cleanup_encoder():
    """Reset encoder singleton before and after each test."""
    reset_sparse_encoder()
    yield
    reset_sparse_encoder()


class TestSparseVectorModel:
    """Tests for the immutable SparseVector domain model."""

    def test_valid_sparse_vector(self):
        """Verify standard initialization with sorted indices and valid weights."""
        sv = SparseVector(indices=(10, 20, 30), values=(0.5, 1.2, 3.0))
        assert sv.indices == (10, 20, 30)
        assert sv.values == (0.5, 1.2, 3.0)
        assert not sv.is_empty
        assert sv.to_dict() == {"indices": [10, 20, 30], "values": [0.5, 1.2, 3.0]}

    def test_empty_sparse_vector(self):
        """Verify empty sparse vector creation and properties."""
        sv = SparseVector()
        assert sv.indices == ()
        assert sv.values == ()
        assert sv.is_empty
        assert sv.to_dict() == {"indices": [], "values": []}

    def test_list_input_converted_to_tuple(self):
        """Verify list arguments are converted to immutable tuples."""
        sv = SparseVector(indices=[5, 15], values=[1.0, 2.0])
        assert isinstance(sv.indices, tuple)
        assert isinstance(sv.values, tuple)
        assert sv.indices == (5, 15)
        assert sv.values == (1.0, 2.0)

    def test_indices_values_length_mismatch_raises(self):
        """Verify error when indices and values lengths differ."""
        with pytest.raises(ValueError, match="length mismatch"):
            SparseVector(indices=(1, 2), values=(1.0,))

    def test_unsorted_or_duplicate_indices_raise(self):
        """Verify error when indices are unsorted or have duplicates."""
        with pytest.raises(ValueError, match="strictly increasing"):
            SparseVector(indices=(20, 10), values=(1.0, 2.0))

        with pytest.raises(ValueError, match="strictly increasing"):
            SparseVector(indices=(10, 10), values=(1.0, 2.0))

    def test_negative_index_raises(self):
        """Verify error when index is negative."""
        with pytest.raises(ValueError, match="non-negative integer"):
            SparseVector(indices=(-1,), values=(1.0,))

    def test_non_finite_weight_raises(self):
        """Verify error when weight is non-finite."""
        with pytest.raises(ValueError, match="finite float"):
            SparseVector(indices=(10,), values=(float("nan"),))

        with pytest.raises(ValueError, match="finite float"):
            SparseVector(indices=(10,), values=(float("inf"),))


class TestTechnicalSparseEncoder:
    """Tests for TechnicalSparseEncoder tokenization, hashing, and weighting."""

    def test_deterministic_encoding(self):
        """Verify that identical text produces strictly identical indices and values."""
        encoder = TechnicalSparseEncoder()
        text = "The user reported ERR-401 when calling API-v2 endpoint."
        sv1 = encoder.encode_document(text)
        sv2 = encoder.encode_document(text)

        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values
        assert not sv1.is_empty

    def test_technical_tokens_preserved(self):
        """Verify that technical identifiers (hyphens, underscores, dots) are preserved."""
        encoder = TechnicalSparseEncoder()
        text = "Check ERR-401 user_id API-v2 C++ BRD-102 voyage-4"
        sv = encoder.encode_document(text)

        tokens = encoder.tokenize(text)
        assert "err-401" in tokens
        assert "user_id" in tokens
        assert "api-v2" in tokens
        assert "c++" in tokens
        assert "brd-102" in tokens
        assert "voyage-4" in tokens

        # Also emits sub-tokens
        assert "err" in tokens
        assert "401" in tokens
        assert "brd" in tokens
        assert "102" in tokens
        assert "user" in tokens
        assert "id" in tokens

    def test_stopwords_filtered_out(self):
        """Verify standard English stopwords are removed unless part of composite."""
        encoder = TechnicalSparseEncoder()
        text = "the and or of to in is that which for with"
        tokens = encoder.tokenize(text)
        assert len(tokens) == 0

        sv = encoder.encode_document(text)
        assert sv.is_empty

    def test_term_frequency_saturation(self):
        """Verify higher term frequency yields higher weight with saturation."""
        encoder = TechnicalSparseEncoder(k1=1.2)
        text_single = "authentication"
        text_triple = "authentication authentication authentication"

        sv_single = encoder.encode_document(text_single)
        sv_triple = encoder.encode_document(text_triple)

        assert len(sv_single.indices) == 1
        assert len(sv_triple.indices) == 1
        assert sv_single.indices == sv_triple.indices
        # 3 occurrences should have strictly higher weight than 1
        assert sv_triple.values[0] > sv_single.values[0]
        # Saturated weight must be <= k1 + 1.0 (2.2)
        assert sv_triple.values[0] < 2.2

    def test_empty_or_whitespace_text(self):
        """Verify empty, whitespace, or punctuation-only text returns empty SparseVector."""
        encoder = TechnicalSparseEncoder()
        assert encoder.encode_document("").is_empty
        assert encoder.encode_document("   \n\t  ").is_empty
        assert encoder.encode_document("!@#$%^&*()").is_empty

    def test_invalid_input_type_raises(self):
        """Verify non-string input raises SparseEncodingError."""
        encoder = TechnicalSparseEncoder()
        with pytest.raises(SparseEncodingError, match="must be a string"):
            encoder.encode_document(12345)  # type: ignore

    def test_batch_encoding_parity(self):
        """Verify encode_documents matches sequential encode_document calls."""
        encoder = TechnicalSparseEncoder()
        texts = [
            "Document indexing error ERR-500",
            "Fast vector search with Qdrant collection",
            "",
            "BRD-102 requirements for RAG pipeline",
        ]
        batch_res = encoder.encode_documents(texts)
        seq_res = [encoder.encode_document(t) for t in texts]

        assert len(batch_res) == len(seq_res)
        for b, s in zip(batch_res, seq_res):
            assert b.indices == s.indices
            assert b.values == s.values

    def test_encode_queries_produces_sparse_vectors(self):
        """Verify encode_queries works identically and produces SparseVectors."""
        encoder = TechnicalSparseEncoder()
        queries = ["how to resolve ERR-401?", "API-v2 latency"]
        results = encoder.encode_queries(queries)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, SparseVector)
            assert not r.is_empty


class TestSparseEncoderFactory:
    """Tests for get_sparse_encoder and registry."""

    def test_get_technical_sparse_encoder(self):
        """Verify factory returns TechnicalSparseEncoder for technical_hash strategy."""
        encoder = get_sparse_encoder("technical_hash")
        assert isinstance(encoder, TechnicalSparseEncoder)
        assert encoder.strategy_name == "technical_hash"
        assert encoder.version == "1.0"

    def test_factory_singleton_caching(self):
        """Verify repeated calls without strategy return the same instance."""
        e1 = get_sparse_encoder("technical_hash")
        e2 = get_sparse_encoder("technical_hash")
        assert e1 is e2

    def test_unknown_strategy_raises(self):
        """Verify unknown strategy name raises SparseEncodingError."""
        with pytest.raises(SparseEncodingError, match="Unknown sparse encoder strategy"):
            get_sparse_encoder("non_existent_strategy")

    def test_reset_singleton(self):
        """Verify reset_sparse_encoder clears the cached instance."""
        e1 = get_sparse_encoder("technical_hash")
        reset_sparse_encoder()
        e2 = get_sparse_encoder("technical_hash")
        assert e1 is not e2
