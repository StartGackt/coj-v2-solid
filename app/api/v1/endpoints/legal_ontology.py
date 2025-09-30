# Placeholder for additional API logic if needed in the future.
from typing import Generator, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.schemas.legal_article import (
    LegalArticleAnalysisError,
    LegalArticleAnalysisRequest,
    LegalArticleAnalysisResponse,
)
from app.schemas.legal_ontology import LegalOntologyCreate, LegalOntologyRead
from app.services.legal_article.factory import build_legal_article_analysis_service
from app.services.legal_article.mapper import map_analysis_to_response
from app.services.legal_ontology_service import LegalOntologyService
from app.services.knowledge_graph import KnowledgeGraphService

router = APIRouter()


@router.get("/legal-ontologies", response_model=List[LegalOntologyRead])
def list_legal_ontologies(db: Session = Depends(get_db)):
    service = LegalOntologyService(LegalOntologyRepository(db))
    return service.get_all()


@router.post(
    "/legal-articles/analyze",
    response_model=LegalArticleAnalysisResponse,
    responses={404: {"model": LegalArticleAnalysisError}},
)
def get_knowledge_graph_service() -> Generator[KnowledgeGraphService, None, None]:
    service = KnowledgeGraphService()
    try:
        yield service
    finally:
        service.close()


@router.post(
    "/legal-articles/analyze",
    response_model=LegalArticleAnalysisResponse,
    responses={404: {"model": LegalArticleAnalysisError}},
)
def analyze_legal_article(
    payload: LegalArticleAnalysisRequest,
    knowledge_graph: KnowledgeGraphService = Depends(get_knowledge_graph_service),
):
    service = build_legal_article_analysis_service(
        knowledge_graph_service=knowledge_graph
    )
    try:
        analysis = service.analyze_article(
            article_number=payload.article_number,
            language=payload.language,
            text_override=payload.text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return map_analysis_to_response(payload.article_number, analysis)


@router.post("/legal-ontologies", response_model=LegalOntologyRead)
def create_legal_ontology(obj_in: LegalOntologyCreate, db: Session = Depends(get_db)):
    service = LegalOntologyService(LegalOntologyRepository(db))
    return service.create(obj_in)


@router.get("/health")
def health_check():
    return {"status": "ok"}
