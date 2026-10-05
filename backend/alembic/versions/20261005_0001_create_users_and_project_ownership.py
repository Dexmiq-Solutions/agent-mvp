"""Create users and refresh_tokens tables, and add user_id ownership to projects table.

Revision ID: 20261005_0001
Revises: 20260914_0002
Create Date: 2026-10-05 11:30:00
"""

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20261005_0001"
down_revision: Union[str, None] = "20260914_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 2. Create refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("token_hash", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_refresh_tokens_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True)
    op.create_index(
        "ix_refresh_tokens_user_id_expires_at",
        "refresh_tokens",
        ["user_id", "expires_at"],
    )

    # 3. Add user_id column to projects table with data migration safety
    op.add_column("projects", sa.Column("user_id", sa.String(length=255), nullable=True))

    # Data migration: backfill any existing unowned projects with a default system owner
    conn = op.get_bind()
    projects_res = conn.execute(sa.text("SELECT count(*) FROM projects WHERE user_id IS NULL")).scalar()
    if projects_res and projects_res > 0:
        system_user_id = str(uuid.uuid4())
        # Insert a disabled placeholder system owner for existing orphaned records
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, hashed_password, is_active, created_at, updated_at) "
                "VALUES (:id, 'system-migrated@agent-mvp.local', 'disabled', false, NOW(), NOW())"
            ),
            {"id": system_user_id},
        )
        conn.execute(
            sa.text("UPDATE projects SET user_id = :system_user_id WHERE user_id IS NULL"),
            {"system_user_id": system_user_id},
        )

    op.alter_column("projects", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_projects_user_id_users",
        "projects",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])
    op.create_index(
        "ix_projects_user_id_created_at",
        "projects",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    # 1. Remove user_id from projects table
    op.drop_index("ix_projects_user_id_created_at", table_name="projects")
    op.drop_index("ix_projects_user_id", table_name="projects")
    op.drop_constraint("fk_projects_user_id_users", "projects", type_="foreignkey")
    op.drop_column("projects", "user_id")

    # 2. Drop refresh_tokens table
    op.drop_index("ix_refresh_tokens_user_id_expires_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    # 3. Drop users table
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
