from typing import Any, Dict, List
from unittest.mock import MagicMock

from app.services.knowledge_graph import KnowledgeGraphService, Neo4jGraphStorage


class StubGraphStorage:
    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.search_calls: List[Dict[str, Any]] = []
        self.closed = False

    def add_node(self, node_id: str, *, label: str = "Entity", properties=None) -> None:
        self.nodes.append(
            {"id": node_id, "label": label, "properties": properties or {}}
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        relationship_type: str = "RELATED_TO",
        source_label: str = "Entity",
        target_label: str = "Entity",
        properties=None,
    ) -> None:
        self.edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "source_label": source_label,
                "target_label": target_label,
                "properties": properties or {},
            }
        )

    def search(self, query: str, *, label: str | None = None, limit: int = 25):
        self.search_calls.append({"query": query, "label": label, "limit": limit})
        return []

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def test_save_entities_and_relationships_calls_storage():
    storage = StubGraphStorage()
    service = KnowledgeGraphService(graph_storage=storage)

    entities = [
        {
            "id": "entity-1",
            "label": "Person",
            "name": "Jane Doe",
            "relationships": [
                {
                    "target": "entity-2",
                    "type": "KNOWS",
                    "target_label": "Person",
                    "weight": 0.8,
                }
            ],
        }
    ]

    service.save_entities_and_relationships(entities)

    assert storage.nodes == [
        {
            "id": "entity-1",
            "label": "Person",
            "properties": {"name": "Jane Doe"},
        }
    ]
    assert storage.edges == [
        {
            "source_id": "entity-1",
            "target_id": "entity-2",
            "relationship_type": "KNOWS",
            "source_label": "Person",
            "target_label": "Person",
            "properties": {"weight": 0.8},
        }
    ]


def test_search_wraps_storage_results():
    storage = StubGraphStorage()
    storage.search = MagicMock(return_value=[{"id": "entity-1"}, {"id": "entity-2"}])
    service = KnowledgeGraphService(graph_storage=storage)

    response = service.search("Jane", limit=10)

    storage.search.assert_called_once_with("Jane", label=None, limit=10)
    assert response == {"count": 2, "results": [{"id": "entity-1"}, {"id": "entity-2"}]}


def _make_storage_with_mocks() -> Neo4jGraphStorage:
    storage: Neo4jGraphStorage = object.__new__(Neo4jGraphStorage)
    storage._uri = "neo4j://127.0.0.1:7687"
    storage._user = None
    storage._password = None
    storage._database = None
    storage._logger = MagicMock()
    storage._driver = MagicMock()
    return storage


def test_neo4j_storage_add_node_sanitises_labels():
    storage = _make_storage_with_mocks()
    storage._execute_write = MagicMock()

    storage.add_node("entity-1", label="Person Profile", properties={"name": "Jane"})

    query = storage._execute_write.call_args[0][0]
    params = storage._execute_write.call_args.kwargs
    assert "PersonProfile" in query  # space removed for Cypher label
    assert params["props"]["id"] == "entity-1"
    assert params["props"]["name"] == "Jane"


def test_neo4j_storage_search_formats_results():
    storage = _make_storage_with_mocks()
    storage._execute_read = MagicMock(
        return_value=[
            {
                "id": "entity-1",
                "labels": ["Entity"],
                "props": {"id": "entity-1", "name": "Jane"},
            }
        ]
    )

    results = storage.search("Jane", label="Person", limit=5)

    query = storage._execute_read.call_args[0][0]
    assert "MATCH (n:Person)" in query
    assert results == [
        {
            "id": "entity-1",
            "labels": ["Entity"],
            "properties": {"id": "entity-1", "name": "Jane"},
        }
    ]
