from typing import Generator

from fastapi import APIRouter, Depends

from app.services.knowledge_graph import KnowledgeGraphService
from app.services.vector_store import VectorStoreService

router = APIRouter()


# Dependency injection for KnowledgeGraphService and VectorStoreService
def get_knowledge_graph_service() -> Generator[KnowledgeGraphService, None, None]:
    service = KnowledgeGraphService()
    try:
        yield service
    finally:
        service.close()


def get_vector_store_service():
    # Replace with actual vector storage implementation
    vector_storage = None  # Example placeholder
    return VectorStoreService(vector_storage)


@router.get("/hybrid-search")
def hybrid_search(
    query: str,
    knowledge_graph_service: KnowledgeGraphService = Depends(
        get_knowledge_graph_service
    ),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Perform a hybrid search using both the Knowledge Graph and Vector Store.

    :param query: The search query.
    :param knowledge_graph_service: Service for Knowledge Graph operations.
    :param vector_store_service: Service for Vector Store operations.
    :return: Combined search results.
    """
    # Example logic for hybrid search (to be replaced with actual implementation)
    graph_results = knowledge_graph_service.search(query)
    vector_results = vector_store_service.search(query)

    # Combine results (example logic)
    combined_results = {
        "graph_results": graph_results,
        "vector_results": vector_results,
    }

    return {"message": "Hybrid search completed", "results": combined_results}
