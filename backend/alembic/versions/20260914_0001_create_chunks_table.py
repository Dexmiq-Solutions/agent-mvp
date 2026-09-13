"""Create chunks table for PostgreSQL Content Store.

Revision ID: 20260914_0001
Revises: None
Create Date: 2026-09-14 02:18:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260914_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.String(length=255), nullable=False),
        sa.Column("document_id", sa.String(length=255), nullable=False),
        sa.Column("document_version_id", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("contextual_content", sa.Text(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("heading", sa.String(length=512), nullable=True),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("section_path", sa.JSON(), nullable=True),
        sa.Column("parent_element_id", sa.String(length=255), nullable=True),
        sa.Column("parent_chunk_id", sa.String(length=255), nullable=True),
        sa.Column("element_types", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_project_id", "chunks", ["project_id"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_document_version_id", "chunks", ["document_version_id"])
    op.create_index("ix_chunks_project_id_chunk_id", "chunks", ["project_id", "chunk_id"])
    op.create_index("ix_chunks_project_id_document_id", "chunks", ["project_id", "document_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_project_id_document_id", table_name="chunks")
    op.drop_index("ix_chunks_project_id_chunk_id", table_name="chunks")
    op.drop_index("ix_chunks_document_version_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("ix_chunks_project_id", table_name="chunks")
    op.drop_table("chunks")
