"""CLI script to re-index documents using the active embedding provider (Cohere Embed v4)."""

import argparse
import asyncio
import sys



from core.config import get_settings
from observability.logging import get_logger
from rag.indexing.reindex import reindex_project_documents

logger = get_logger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-index documents into Qdrant using active Cohere Embed v4 provider."
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="Project ID to re-index all READY documents for.",
    )
    args = parser.parse_args()

    settings = get_settings()
    logger.info(
        "Starting re-indexing for project '%s' using provider '%s' (model: '%s')",
        args.project_id,
        settings.EMBEDDING_PROVIDER,
        settings.EMBEDDING_MODEL,
    )

    reports = await reindex_project_documents(project_id=args.project_id, settings=settings)
    total_indexed = sum(r.total_indexed for r in reports)
    logger.info(
        "Re-indexing complete for project '%s': %d document versions processed, %d vector points indexed.",
        args.project_id,
        len(reports),
        total_indexed,
    )


if __name__ == "__main__":
    asyncio.run(main())
