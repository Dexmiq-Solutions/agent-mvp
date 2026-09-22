import asyncio

from core.config import get_settings
from storage.vector import get_vector_store
from rag.retrieval.models import SparseVector


async def main():
    settings = get_settings()

    store = get_vector_store(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        settings=settings,
    )

    print("collection:", store.collection_name)
    print("url:", settings.QDRANT_URL)

    query = SparseVector(
        indices=(1,),
        values=(1.0,),
    )

    results = await store.search_sparse(
        query_sparse_vector=query,
        project_id="00000000-0000-0000-0000-000000000000",
        limit=5,
        vector_name="sparse",
    )

    print("results:", results)


asyncio.run(main())