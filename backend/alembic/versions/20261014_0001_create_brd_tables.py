"""Create canonical BRD, version snapshots, and structured sections tables.

Revision ID: 20261014_0001
Revises: 20261005_0001
Create Date: 2026-10-14 15:45:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.types import JSON

# revision identifiers, used by Alembic.
revision: str = "20261014_0001"
down_revision: Union[str, None] = "20260914_0002" if False else "20261005_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create canonical brds table (1:1 with projects)
    op.create_table(
        "brds",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_brds_project_id_projects",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_brds_project_id"),
    )
    op.create_index("ix_brds_project_id", "brds", ["project_id"], unique=True)

    # 2. Create brd_versions table (drafts & historical snapshots)
    op.create_table(
        "brd_versions",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("brd_id", sa.String(length=255), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("version_label", sa.String(length=50), nullable=False, server_default="0.1"),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="draft"),
        sa.Column("assembled_markdown", sa.Text(), nullable=True),
        sa.Column("metadata", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["brd_id"],
            ["brds.id"],
            name="fk_brd_versions_brd_id_brds",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brd_id", "version_number", name="uq_brd_versions_brd_id_version_number"),
    )
    op.create_index("ix_brd_versions_brd_id", "brd_versions", ["brd_id"])
    op.create_index("ix_brd_versions_status", "brd_versions", ["status"])
    op.create_index("ix_brd_versions_brd_id_status", "brd_versions", ["brd_id", "status"])
    op.create_index(
        "uq_brd_versions_one_current",
        "brd_versions",
        ["brd_id"],
        unique=True,
        postgresql_where=sa.text("is_current = true"),
    )

    # 3. Create brd_sections table (granular structured sections)
    op.create_table(
        "brd_sections",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("brd_version_id", sa.String(length=255), nullable=False),
        sa.Column("section_key", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("heading_level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("parent_section_id", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="Not Started"),
        sa.Column("rework_feedback", sa.Text(), nullable=True),
        sa.Column("metadata", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["brd_version_id"],
            ["brd_versions.id"],
            name="fk_brd_sections_brd_version_id_brd_versions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_section_id"],
            ["brd_sections.id"],
            name="fk_brd_sections_parent_section_id_brd_sections",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brd_version_id", "section_key", name="uq_brd_sections_version_id_section_key"),
    )
    op.create_index("ix_brd_sections_brd_version_id", "brd_sections", ["brd_version_id"])
    op.create_index("ix_brd_sections_version_order", "brd_sections", ["brd_version_id", "order_index"])


def downgrade() -> None:
    # 1. Drop brd_sections table
    op.drop_index("ix_brd_sections_version_order", table_name="brd_sections")
    op.drop_index("ix_brd_sections_brd_version_id", table_name="brd_sections")
    op.drop_table("brd_sections")

    # 2. Drop brd_versions table
    op.drop_index("uq_brd_versions_one_current", table_name="brd_versions", postgresql_where=sa.text("is_current = true"))
    op.drop_index("ix_brd_versions_brd_id_status", table_name="brd_versions")
    op.drop_index("ix_brd_versions_status", table_name="brd_versions")
    op.drop_index("ix_brd_versions_brd_id", table_name="brd_versions")
    op.drop_table("brd_versions")

    # 3. Drop brds table
    op.drop_index("ix_brds_project_id", table_name="brds")
    op.drop_table("brds")
