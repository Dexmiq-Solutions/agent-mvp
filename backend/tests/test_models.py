"""Unit and integration tests for the relational application data model.

Tests coverage:
- Model registration in Base.metadata
- Table definitions, column types, nullability, defaults
- Foreign keys and unique constraints
- Relational mapping (Project -> Documents -> Versions -> Chunks, Project -> Conversations -> Messages)
- Database-level cascade deletes and integrity constraints
- Compatibility and regression checks for ChunkModel
- Alembic migration definition and revision chain
"""

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import models
from models import (
    Base,
    ChunkModel,
    ConversationModel,
    DocumentModel,
    DocumentVersionModel,
    DocumentVersionStatus,
    MessageModel,
    ProjectModel,
)


# ==============================================================================
# 1. Model Registration & Export Tests
# ==============================================================================


def test_model_registration_in_metadata():
    """Verify all 6 core tables are registered with Base.metadata."""
    table_names = set(Base.metadata.tables.keys())
    expected_tables = {
        "projects",
        "documents",
        "document_versions",
        "chunks",
        "conversations",
        "messages",
    }
    assert expected_tables.issubset(table_names), f"Missing tables: {expected_tables - table_names}"


def test_models_package_exports():
    """Verify models.__init__.py exports all application entities and statuses."""
    expected_symbols = [
        "Base",
        "ProjectModel",
        "DocumentModel",
        "DocumentVersionModel",
        "DocumentVersionStatus",
        "ConversationModel",
        "MessageModel",
        "ChunkModel",
    ]
    for sym in expected_symbols:
        assert hasattr(models, sym), f"models package missing symbol '{sym}'"
        assert sym in models.__all__, f"'{sym}' not declared in models.__all__"


# ==============================================================================
# 2. Table Structure & Column Integrity Tests
# ==============================================================================


def test_project_model_structure():
    """Verify ProjectModel table definition, columns, and serialization."""
    table = Base.metadata.tables["projects"]

    assert table.name == "projects"
    assert "id" in table.c
    assert table.c.id.primary_key
    assert "name" in table.c
    assert not table.c.name.nullable
    assert "description" in table.c
    assert table.c.description.nullable
    assert "created_at" in table.c
    assert "updated_at" in table.c

    proj = ProjectModel(id="proj_1", name="Alpha Project", description="Test project description")
    assert proj.id == "proj_1"
    assert proj.name == "Alpha Project"
    assert "ProjectModel(id='proj_1', name='Alpha Project')" in repr(proj)

    d = proj.to_dict()
    assert d["id"] == "proj_1"
    assert d["name"] == "Alpha Project"
    assert d["description"] == "Test project description"


def test_document_model_structure():
    """Verify DocumentModel table definition, columns, foreign keys, and indexes."""
    table = Base.metadata.tables["documents"]

    assert table.name == "documents"
    assert "id" in table.c
    assert table.c.id.primary_key
    assert "project_id" in table.c
    assert not table.c.project_id.nullable
    assert "name" in table.c
    assert not table.c.name.nullable
    assert "created_at" in table.c
    assert "updated_at" in table.c

    # Foreign key verification
    fk_targets = [fk.target_fullname for fk in table.c.project_id.foreign_keys]
    assert "projects.id" in fk_targets

    # Index verification
    index_names = {idx.name for idx in table.indexes}
    assert "ix_documents_project_id" in index_names
    assert "ix_documents_project_id_created_at" in index_names

    doc = DocumentModel(id="doc_1", project_id="proj_1", name="Architecture.pdf")
    assert doc.id == "doc_1"
    assert doc.project_id == "proj_1"
    assert "DocumentModel(id='doc_1', project_id='proj_1', name='Architecture.pdf')" in repr(doc)

    d = doc.to_dict()
    assert d["id"] == "doc_1"
    assert d["project_id"] == "proj_1"
    assert d["name"] == "Architecture.pdf"


def test_document_version_model_structure():
    """Verify DocumentVersionModel table definition, unique constraint, foreign keys, and status enum."""
    table = Base.metadata.tables["document_versions"]

    assert table.name == "document_versions"
    assert "id" in table.c
    assert table.c.id.primary_key
    assert "document_id" in table.c
    assert not table.c.document_id.nullable
    assert "project_id" in table.c
    assert not table.c.project_id.nullable
    assert "version_number" in table.c
    assert not table.c.version_number.nullable
    assert "storage_bucket" in table.c
    assert "storage_path" in table.c
    assert "original_filename" in table.c
    assert "content_type" in table.c
    assert "size_bytes" in table.c
    assert "etag" in table.c
    assert "status" in table.c
    assert "error_message" in table.c
    assert "indexed_at" in table.c
    assert "created_at" in table.c
    assert "updated_at" in table.c

    # Foreign keys
    doc_fk_targets = [fk.target_fullname for fk in table.c.document_id.foreign_keys]
    assert "documents.id" in doc_fk_targets
    proj_fk_targets = [fk.target_fullname for fk in table.c.project_id.foreign_keys]
    assert "projects.id" in proj_fk_targets

    # Unique constraint (document_id, version_number)
    uq_constraints = [
        c for c in table.constraints if isinstance(c, sa.UniqueConstraint)
    ]
    uq_col_sets = [{col.name for col in c.columns} for c in uq_constraints]
    assert {"document_id", "version_number"} in uq_col_sets

    # Status Enum
    assert DocumentVersionStatus.PENDING.value == "pending"
    assert DocumentVersionStatus.INDEXING.value == "indexing"
    assert DocumentVersionStatus.READY.value == "ready"
    assert DocumentVersionStatus.FAILED.value == "failed"

    ver = DocumentVersionModel(
        id="ver_1",
        document_id="doc_1",
        project_id="proj_1",
        version_number=1,
        storage_bucket="documents",
        storage_path="proj_1/Architecture.pdf",
        original_filename="Architecture.pdf",
        content_type="application/pdf",
        size_bytes=10240,
        etag="etag_xyz",
        status=DocumentVersionStatus.READY.value,
    )
    assert ver.version_number == 1
    assert ver.status == "ready"
    assert "DocumentVersionModel(id='ver_1', document_id='doc_1', version_number=1, status='ready')" in repr(ver)

    d = ver.to_dict()
    assert d["id"] == "ver_1"
    assert d["document_id"] == "doc_1"
    assert d["project_id"] == "proj_1"
    assert d["storage_bucket"] == "documents"
    assert d["status"] == "ready"


def test_conversation_model_structure():
    """Verify ConversationModel table definition, foreign keys, and indexes."""
    table = Base.metadata.tables["conversations"]

    assert table.name == "conversations"
    assert "id" in table.c
    assert table.c.id.primary_key
    assert "project_id" in table.c
    assert not table.c.project_id.nullable
    assert "title" in table.c
    assert "created_at" in table.c
    assert "updated_at" in table.c

    proj_fk_targets = [fk.target_fullname for fk in table.c.project_id.foreign_keys]
    assert "projects.id" in proj_fk_targets

    index_names = {idx.name for idx in table.indexes}
    assert "ix_conversations_project_id" in index_names
    assert "ix_conversations_project_id_updated_at" in index_names

    conv = ConversationModel(id="conv_1", project_id="proj_1", title="Initial chat")
    assert conv.id == "conv_1"
    assert conv.title == "Initial chat"
    assert "ConversationModel(id='conv_1', project_id='proj_1', title='Initial chat')" in repr(conv)

    d = conv.to_dict()
    assert d["id"] == "conv_1"
    assert d["project_id"] == "proj_1"
    assert d["title"] == "Initial chat"


def test_message_model_structure():
    """Verify MessageModel table definition, foreign keys, indexes, and metadata handling."""
    table = Base.metadata.tables["messages"]

    assert table.name == "messages"
    assert "id" in table.c
    assert table.c.id.primary_key
    assert "conversation_id" in table.c
    assert not table.c.conversation_id.nullable
    assert "role" in table.c
    assert not table.c.role.nullable
    assert "content" in table.c
    assert not table.c.content.nullable
    assert "metadata" in table.c
    assert "created_at" in table.c

    conv_fk_targets = [fk.target_fullname for fk in table.c.conversation_id.foreign_keys]
    assert "conversations.id" in conv_fk_targets

    index_names = {idx.name for idx in table.indexes}
    assert "ix_messages_conversation_id" in index_names
    assert "ix_messages_conversation_id_created_at" in index_names

    msg = MessageModel(
        id="msg_1",
        conversation_id="conv_1",
        role="user",
        content="What is the system architecture?",
        message_metadata={"tokens": 12},
    )
    assert msg.role == "user"
    assert msg.meta == {"tokens": 12}
    assert "MessageModel(id='msg_1', conversation_id='conv_1', role='user'" in repr(msg)

    d = msg.to_dict()
    assert d["id"] == "msg_1"
    assert d["conversation_id"] == "conv_1"
    assert d["role"] == "user"
    assert d["content"] == "What is the system architecture?"
    assert d["metadata"] == {"tokens": 12}


def test_chunk_model_preservation_and_foreign_keys():
    """Verify ChunkModel preserves all existing attributes, properties, and gains foreign keys."""
    table = Base.metadata.tables["chunks"]

    # Preserved columns
    required_cols = [
        "chunk_id",
        "project_id",
        "document_id",
        "document_version_id",
        "content",
        "contextual_content",
        "chunk_index",
        "heading",
        "heading_level",
        "section_path",
        "parent_element_id",
        "parent_chunk_id",
        "element_types",
        "metadata",
        "created_at",
        "updated_at",
    ]
    for col in required_cols:
        assert col in table.c, f"ChunkModel table missing column: {col}"

    # Nullability preservation
    assert not table.c.chunk_id.nullable
    assert not table.c.project_id.nullable
    assert not table.c.document_id.nullable
    assert table.c.document_version_id.nullable, "document_version_id must remain nullable"

    # Foreign keys
    proj_fk_targets = [fk.target_fullname for fk in table.c.project_id.foreign_keys]
    assert "projects.id" in proj_fk_targets

    doc_fk_targets = [fk.target_fullname for fk in table.c.document_id.foreign_keys]
    assert "documents.id" in doc_fk_targets

    ver_fk_targets = [fk.target_fullname for fk in table.c.document_version_id.foreign_keys]
    assert "document_versions.id" in ver_fk_targets

    # Compatibility properties
    chunk = ChunkModel(
        chunk_id="chunk_1",
        project_id="proj_1",
        document_id="doc_1",
        document_version_id="ver_1",
        content="verbatim content",
        contextual_content="enriched contextual content",
        chunk_index=3,
        heading="Introduction",
        heading_level=1,
        section_path=["Root", "Intro"],
        chunk_metadata={"token_count": 42},
    )
    assert chunk.text == "verbatim content"
    assert chunk.chunk_text == "verbatim content"
    assert chunk.index == 3
    assert chunk.meta == {"token_count": 42}

    d = chunk.to_dict()
    assert d["chunk_id"] == "chunk_1"
    assert d["content"] == "verbatim content"
    assert d["chunk_text"] == "verbatim content"
    assert d["index"] == 3
    assert d["heading"] == "Introduction"
    assert d["metadata"] == {"token_count": 42}


# ==============================================================================
# 3. In-Memory Relationship Mapping Tests
# ==============================================================================


def test_hierarchical_relationships_in_memory():
    """Verify in-memory object graph navigation across all relationships."""
    proj = ProjectModel(id="p1", name="Project 1")
    doc = DocumentModel(id="d1", name="Doc 1")
    ver = DocumentVersionModel(
        id="v1",
        version_number=1,
        storage_bucket="b",
        storage_path="p",
        original_filename="f",
    )
    chunk = ChunkModel(chunk_id="c1", content="chunk text")

    # Wire Project -> Document -> Version -> Chunk
    proj.documents.append(doc)
    doc.versions.append(ver)
    ver.chunks.append(chunk)

    assert doc.project is proj
    assert ver.document is doc
    assert chunk.document_version is ver
    assert len(proj.documents) == 1
    assert len(doc.versions) == 1
    assert len(ver.chunks) == 1

    # Wire Project -> Conversation -> Message
    conv = ConversationModel(id="conv1", title="Conversation 1")
    msg1 = MessageModel(id="m1", role="user", content="Hello")
    msg2 = MessageModel(id="m2", role="assistant", content="Hi there!")

    proj.conversations.append(conv)
    conv.messages.extend([msg1, msg2])

    assert conv.project is proj
    assert msg1.conversation is conv
    assert msg2.conversation is conv
    assert len(proj.conversations) == 1
    assert len(conv.messages) == 2


# ==============================================================================
# 4. Database-Level Relational Cascade & Integrity Tests (SQLite in-memory)
# ==============================================================================


@pytest.fixture
def memory_db():
    """Set up an in-memory SQLite database with foreign keys strictly enabled."""
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_cascade_deletion_from_project_boundary(memory_db: Session):
    """Verify that deleting a Project cascades and removes all dependent documents, versions, chunks, conversations, and messages."""
    proj = ProjectModel(id="p_root", name="Root Project")
    doc = DocumentModel(id="d_child", project_id="p_root", name="Doc Child")
    ver = DocumentVersionModel(
        id="v_child",
        document_id="d_child",
        project_id="p_root",
        version_number=1,
        storage_bucket="bucket",
        storage_path="path/to/file",
        original_filename="file.pdf",
    )
    chunk = ChunkModel(
        chunk_id="c_child",
        project_id="p_root",
        document_id="d_child",
        document_version_id="v_child",
        content="cascade test content",
    )
    conv = ConversationModel(id="conv_child", project_id="p_root", title="Conv Child")
    msg = MessageModel(id="m_child", conversation_id="conv_child", role="user", content="Test")

    memory_db.add_all([proj, doc, ver, chunk, conv, msg])
    memory_db.commit()

    assert memory_db.query(ProjectModel).count() == 1
    assert memory_db.query(DocumentModel).count() == 1
    assert memory_db.query(DocumentVersionModel).count() == 1
    assert memory_db.query(ChunkModel).count() == 1
    assert memory_db.query(ConversationModel).count() == 1
    assert memory_db.query(MessageModel).count() == 1

    # Delete project
    memory_db.delete(proj)
    memory_db.commit()

    # Verify complete cascade deletion
    assert memory_db.query(ProjectModel).count() == 0
    assert memory_db.query(DocumentModel).count() == 0
    assert memory_db.query(DocumentVersionModel).count() == 0
    assert memory_db.query(ChunkModel).count() == 0
    assert memory_db.query(ConversationModel).count() == 0
    assert memory_db.query(MessageModel).count() == 0


def test_unique_constraint_document_version_number(memory_db: Session):
    """Verify unique constraint prevents duplicate version numbers for the same document."""
    proj = ProjectModel(id="p_uq", name="UQ Project")
    doc = DocumentModel(id="d_uq", project_id="p_uq", name="UQ Doc")
    ver1 = DocumentVersionModel(
        id="v1_uq",
        document_id="d_uq",
        project_id="p_uq",
        version_number=1,
        storage_bucket="bucket",
        storage_path="path/v1",
        original_filename="doc.pdf",
    )
    memory_db.add_all([proj, doc, ver1])
    memory_db.commit()

    # Attempt inserting duplicate version_number 1 for same document
    ver2_duplicate = DocumentVersionModel(
        id="v2_duplicate",
        document_id="d_uq",
        project_id="p_uq",
        version_number=1,
        storage_bucket="bucket",
        storage_path="path/v2",
        original_filename="doc.pdf",
    )
    memory_db.add(ver2_duplicate)
    with pytest.raises(IntegrityError):
        memory_db.commit()
    memory_db.rollback()


def test_foreign_key_enforcement(memory_db: Session):
    """Verify that referencing a non-existent parent raises an IntegrityError."""
    orphan_doc = DocumentModel(id="d_orphan", project_id="non_existent_proj", name="Orphan")
    memory_db.add(orphan_doc)
    with pytest.raises(IntegrityError):
        memory_db.commit()
    memory_db.rollback()


# ==============================================================================
# 5. Alembic Migration & Revision Chain Tests
# ==============================================================================


def test_alembic_revision_chain():
    """Verify Alembic migration 20260914_0002 correctly chains to 20260914_0001."""
    backend_dir = Path(__file__).resolve().parents[1]
    ini_path = backend_dir / "alembic.ini"
    assert ini_path.exists()

    cfg = Config(str(ini_path))
    script = ScriptDirectory.from_config(cfg)
    revisions = list(script.walk_revisions())

    rev_map = {r.revision: r.down_revision for r in revisions}
    assert "20260914_0002" in rev_map, "Migration 20260914_0002 not found in Alembic script directory"
    assert "20260914_0001" in rev_map, "Base migration 20260914_0001 not found in Alembic script directory"
    assert rev_map["20260914_0002"] == "20260914_0001", "Migration 20260914_0002 must have down_revision '20260914_0001'"
    assert rev_map["20260914_0001"] is None, "Base migration 20260914_0001 must have None as down_revision"


def test_alembic_migration_module_callables():
    """Verify migration module exports upgrade() and downgrade() functions."""
    migration_file = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "20260914_0002_create_application_tables.py"
    )
    assert migration_file.exists()

    spec = importlib.util.spec_from_file_location("migration_0002", migration_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "upgrade")
    assert callable(module.upgrade)
    assert hasattr(module, "downgrade")
    assert callable(module.downgrade)
    assert module.revision == "20260914_0002"
    assert module.down_revision == "20260914_0001"


def test_alembic_offline_sql_upgrade_and_downgrade(monkeypatch):
    """Verify offline SQL generation runs without error for both upgrade and downgrade."""
    import io
    from contextlib import redirect_stdout
    from alembic import command
    from app.core.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")
    get_settings.cache_clear()
    try:
        backend_dir = Path(__file__).resolve().parents[1]
        cfg = Config(str(backend_dir / "alembic.ini"))

        # Test upgrade --sql
        buf = io.StringIO()
        with redirect_stdout(buf):
            command.upgrade(cfg, "20260914_0002", sql=True)
        sql_upgrade = buf.getvalue()
        assert "CREATE TABLE projects" in sql_upgrade
        assert "CREATE TABLE documents" in sql_upgrade
        assert "CREATE TABLE document_versions" in sql_upgrade
        assert "CREATE TABLE conversations" in sql_upgrade
        assert "CREATE TABLE messages" in sql_upgrade
        assert "fk_chunks_project_id_projects" in sql_upgrade

        # Test downgrade --sql
        buf_down = io.StringIO()
        with redirect_stdout(buf_down):
            command.downgrade(cfg, "20260914_0002:20260914_0001", sql=True)
        sql_downgrade = buf_down.getvalue()
        assert "DROP TABLE messages" in sql_downgrade
        assert "DROP TABLE conversations" in sql_downgrade
        assert "DROP TABLE document_versions" in sql_downgrade
        assert "DROP TABLE documents" in sql_downgrade
        assert "DROP TABLE projects" in sql_downgrade
        assert "fk_chunks_project_id_projects" in sql_downgrade
    finally:
        get_settings.cache_clear()
