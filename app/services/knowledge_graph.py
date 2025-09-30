"""Knowledge graph service powered by Neo4j."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


class Neo4jGraphStorage:
    """Thin wrapper around the Neo4j driver providing graph operations."""

    def __init__(
        self,
        uri: str,
        *,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._logger = logger or LOGGER
        self._driver: Driver = self._create_driver()

    # ---------------------------------------------------------------------
    # lifecycle helpers
    # ---------------------------------------------------------------------
    def _create_driver(self) -> Driver:
        auth = None
        if self._user and self._password:
            auth = (self._user, self._password)

        try:
            driver = GraphDatabase.driver(self._uri, auth=auth)
            # Verify connectivity eagerly to fail fast during startup.
            driver.verify_connectivity()
            return driver
        except AttributeError:
            driver = GraphDatabase.driver(self._uri, auth=auth)
            with driver.session(database=self._database) as session:
                session.run("RETURN 1").consume()
            return driver
        except AuthError as exc:  # pragma: no cover - relies on external Neo4j instance
            self._logger.error("Neo4j authentication failed: %s", exc)
            raise
        except ServiceUnavailable as exc:  # pragma: no cover - network failure
            self._logger.error("Neo4j is unavailable at %s: %s", self._uri, exc)
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def __del__(self) -> None:  # pragma: no cover - non deterministic
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # public operations
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        *,
        label: str = "Entity",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_label = _sanitize_label(label)
        props = _sanitize_properties(properties or {})
        props["id"] = node_id

        query = (
            f"MERGE (n:{safe_label} {{id: $node_id}}) "
            "SET n += $props, n.updated_at = datetime()"
        )

        self._execute_write(query, node_id=node_id, props=props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        relationship_type: str = "RELATED_TO",
        source_label: str = "Entity",
        target_label: str = "Entity",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_rel = _sanitize_label(relationship_type)
        safe_source_label = _sanitize_label(source_label)
        safe_target_label = _sanitize_label(target_label)
        props = _sanitize_properties(properties or {})

        query = (
            f"MATCH (a:{safe_source_label} {{id: $source_id}}), "
            f"(b:{safe_target_label} {{id: $target_id}}) "
            f"MERGE (a)-[r:{safe_rel}]->(b) "
            "SET r += $props, r.updated_at = datetime()"
        )

        self._execute_write(
            query,
            source_id=source_id,
            target_id=target_id,
            props=props,
        )

    def search(
        self,
        query: str,
        *,
        label: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        safe_label = f":{_sanitize_label(label)}" if label else ""

        cypher = (
            f"MATCH (n{safe_label}) "
            "WHERE any(key IN keys(n) "
            "WHERE toLower(toString(n[key])) CONTAINS toLower($query)) "
            "RETURN labels(n) AS labels, n.id AS id, properties(n) AS props "
            "LIMIT $limit"
        )

        records = self._execute_read(cypher, query=query, limit=limit)
        results: List[Dict[str, Any]] = []
        for record in records:
            node_props: Dict[str, Any] = dict(record.get("props", {}))
            results.append(
                {
                    "id": record.get("id"),
                    "labels": list(record.get("labels", [])),
                    "properties": _deserialize_properties(node_props),
                }
            )
        return results

    def health_check(self) -> bool:
        try:
            self._execute_read("RETURN 1 AS ok")
            return True
        except (ServiceUnavailable, Neo4jError):
            return False

    # ------------------------------------------------------------------
    # execution helpers
    # ------------------------------------------------------------------
    def _execute_write(self, query: str, **parameters: Any) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(lambda tx: tx.run(query, **parameters).consume())
        except Neo4jError as exc:
            self._logger.exception("Neo4j write failed: %s", exc)
            raise

    def _execute_read(self, query: str, **parameters: Any) -> Iterable[Dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.execute_read(lambda tx: tx.run(query, **parameters))
                return list(result)
        except Neo4jError as exc:
            self._logger.exception("Neo4j read failed: %s", exc)
            raise


class KnowledgeGraphService:
    """Facade exposing business-level graph operations."""

    def __init__(self, graph_storage: Optional[Neo4jGraphStorage] = None) -> None:
        if graph_storage is None:
            graph_storage = Neo4jGraphStorage(
                settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE,
            )
        self.graph_storage = graph_storage

    def save_entities_and_relationships(
        self, entities: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        for entity in entities:
            node_id = _coalesce_entity_id(entity)
            label = entity.get("label") or entity.get("entity") or "Entity"

            properties = {
                key: value
                for key, value in entity.items()
                if key not in {"id", "entity", "label", "relationships"}
            }

            self.graph_storage.add_node(
                node_id,
                label=label,
                properties=properties,
            )

            for relationship in entity.get("relationships", []):
                target_id = _coalesce_entity_id(relationship, fallback_key="target")
                rel_type = relationship.get("type") or "RELATED_TO"
                source_label = label
                target_label = (
                    relationship.get("target_label")
                    or relationship.get("label")
                    or "Entity"
                )

                rel_properties = {
                    key: value
                    for key, value in relationship.items()
                    if key
                    not in {
                        "target",
                        "type",
                        "label",
                        "target_label",
                        "source_label",
                        "id",
                    }
                }

                self.graph_storage.add_edge(
                    node_id,
                    target_id,
                    relationship_type=rel_type,
                    source_label=relationship.get("source_label", source_label),
                    target_label=target_label,
                    properties=rel_properties,
                )

        return {
            "status": "success",
            "message": "Entities and relationships saved to Knowledge Graph.",
        }

    def search(
        self, query: str, *, label: Optional[str] = None, limit: int = 25
    ) -> Dict[str, Any]:
        records = self.graph_storage.search(query, label=label, limit=limit)
        return {"count": len(records), "results": records}

    def health_check(self) -> bool:
        return self.graph_storage.health_check()

    def close(self) -> None:
        self.graph_storage.close()


def _coalesce_entity_id(data: Dict[str, Any], *, fallback_key: str = "id") -> str:
    for key in (fallback_key, "id", "entity", "name"):
        value = data.get(key)
        if value:
            return str(value)
    raise ValueError("Entity data must include an identifier field")


def _sanitize_label(label: Optional[str]) -> str:
    if not label:
        return "Entity"
    allowed = {"_"}.union({chr(code) for code in range(ord("0"), ord("9") + 1)})
    allowed.update(chr(code) for code in range(ord("A"), ord("Z") + 1))
    allowed.update(chr(code) for code in range(ord("a"), ord("z") + 1))
    cleaned = "".join(char for char in label if char in allowed)
    if not cleaned:
        raise ValueError(f"Invalid Cypher label: {label}")
    return cleaned


def _sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, (dict, list)):
            sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


def _deserialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    return properties
