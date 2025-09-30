from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from app.core.contracts.legal_article import (
    ComplianceStepDetail,
    LegalArticle,
    LegalArticleAnalysis,
    LegalArticleAnalyzerProtocol,
    LegalArticleRepositoryProtocol,
    ObligationDetail,
)
from app.services.knowledge_graph import KnowledgeGraphService


LOGGER = logging.getLogger(__name__)


class LegalArticleAnalysisService:
    """Coordinates retrieval and analysis of legal articles."""

    def __init__(
        self,
        repository: LegalArticleRepositoryProtocol,
        analyzer: LegalArticleAnalyzerProtocol,
        *,
        knowledge_graph: KnowledgeGraphService | None = None,
    ) -> None:
        self._repository = repository
        self._analyzer = analyzer
        self._knowledge_graph = knowledge_graph

    def analyze_article(
        self,
        article_number: str,
        language: str,
        text_override: Optional[str] = None,
    ) -> LegalArticleAnalysis:
        article = self._resolve_article(article_number, language, text_override)
        analysis = self._analyzer.analyze(article)
        self._persist_to_knowledge_graph(article, analysis)
        return analysis

    def _resolve_article(
        self, article_number: str, language: str, text_override: Optional[str]
    ) -> LegalArticle:
        if text_override:
            return LegalArticle(
                number=article_number, language=language, text=text_override
            )

        stored_article = self._repository.get_article(article_number, language)
        if stored_article:
            return stored_article

        raise ValueError(
            f"Legal article '{article_number}' in language '{language}' is not available and no text override was provided."
        )

    def _persist_to_knowledge_graph(
        self, article: LegalArticle, analysis: LegalArticleAnalysis
    ) -> None:
        if not self._knowledge_graph:
            return

        try:
            entities = _build_graph_entities(article, analysis)
            if entities:
                self._knowledge_graph.save_entities_and_relationships(entities)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Failed to persist article %s to knowledge graph: %s",
                article.number,
                exc,
            )


def _build_graph_entities(
    article: LegalArticle, analysis: LegalArticleAnalysis
) -> List[dict]:
    article_id = _slugify(f"{article.number}_{article.language}")
    base_entity = {
        "id": f"article::{article_id}",
        "label": "LegalArticle",
        "article_number": article.number,
        "language": article.language,
        "summary": analysis.summary,
        "relationships": [],
    }

    entities: List[dict] = [base_entity]

    _attach_obligations(base_entity, entities, article_id, analysis.obligations)
    _attach_exceptions(base_entity, entities, article_id, analysis.exceptions)
    _attach_timelines(base_entity, entities, article_id, analysis.timelines)
    _attach_compliance_steps(
        base_entity, entities, article_id, analysis.compliance_steps
    )

    return entities


def _attach_obligations(
    base_entity: dict,
    entities: List[dict],
    article_id: str,
    obligations: Iterable[ObligationDetail],
) -> None:
    for index, obligation in enumerate(obligations, start=1):
        obligation_id = f"{article_id}::obligation::{index}"
        entities.append(
            {
                "id": obligation_id,
                "label": "LegalObligation",
                "actor": obligation.actor,
                "action": obligation.action,
                "timeline": obligation.timeline,
            }
        )

        relationship = {
            "target": obligation_id,
            "type": "HAS_OBLIGATION",
            "target_label": "LegalObligation",
            "actor": obligation.actor,
        }
        if obligation.timeline:
            relationship["timeline"] = obligation.timeline

        base_entity.setdefault("relationships", []).append(relationship)


def _attach_exceptions(
    base_entity: dict,
    entities: List[dict],
    article_id: str,
    exceptions: Iterable[str],
) -> None:
    for index, exception in enumerate(exceptions, start=1):
        exception_id = f"{article_id}::exception::{index}"
        entities.append(
            {
                "id": exception_id,
                "label": "LegalException",
                "description": exception,
            }
        )
        base_entity.setdefault("relationships", []).append(
            {
                "target": exception_id,
                "type": "HAS_EXCEPTION",
                "target_label": "LegalException",
            }
        )


def _attach_timelines(
    base_entity: dict,
    entities: List[dict],
    article_id: str,
    timelines: Iterable[str],
) -> None:
    for index, timeline in enumerate(timelines, start=1):
        timeline_id = f"{article_id}::timeline::{index}"
        entities.append(
            {
                "id": timeline_id,
                "label": "LegalTimeline",
                "description": timeline,
            }
        )
        base_entity.setdefault("relationships", []).append(
            {
                "target": timeline_id,
                "type": "HAS_TIMELINE",
                "target_label": "LegalTimeline",
            }
        )


def _attach_compliance_steps(
    base_entity: dict,
    entities: List[dict],
    article_id: str,
    steps: Iterable[ComplianceStepDetail],
) -> None:
    for index, step in enumerate(steps, start=1):
        step_id = f"{article_id}::compliance_step::{index}"
        entities.append(
            {
                "id": step_id,
                "label": "ComplianceStep",
                "description": step.description,
                "rationale": step.rationale,
                "order": index,
            }
        )
        base_entity.setdefault("relationships", []).append(
            {
                "target": step_id,
                "type": "HAS_COMPLIANCE_STEP",
                "target_label": "ComplianceStep",
                "order": index,
            }
        )


def _slugify(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    return "".join(cleaned).strip("_")
