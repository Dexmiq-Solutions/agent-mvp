"""Add workflow selection and execution lease columns to conversations table.

Revision ID: 20261016_0001
Revises: 20261014_0001
Create Date: 2026-10-16 10:00:00
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20261016_0001"
down_revision: Union[str, None] = "20261014_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("active_workflow", sa.String(length=50), nullable=False, server_default="brd"),
    )
    op.add_column(
        "conversations",
        sa.Column("execution_status", sa.String(length=50), nullable=False, server_default="idle"),
    )
    op.add_column(
        "conversations",
        sa.Column("active_run_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_conversations_lease_expires_at",
        "conversations",
        ["lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_lease_expires_at", table_name="conversations")
    op.drop_column("conversations", "lease_expires_at")
    op.drop_column("conversations", "active_run_id")
    op.drop_column("conversations", "execution_status")
    op.drop_column("conversations", "active_workflow")
