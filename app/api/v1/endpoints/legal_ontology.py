# Placeholder for additional API logic if needed in the future.
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.legal_ontology import LegalOntologyCreate, LegalOntologyRead
from app.services.legal_ontology_service import LegalOntologyService
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.api.dependencies import get_db
from typing import List

router = APIRouter()


@router.get("/legal-ontologies", response_model=List[LegalOntologyRead])
def list_legal_ontologies(db: Session = Depends(get_db)):
    service = LegalOntologyService(LegalOntologyRepository(db))
    return service.get_all()


@router.post("/legal-ontologies", response_model=LegalOntologyRead)
def create_legal_ontology(obj_in: LegalOntologyCreate, db: Session = Depends(get_db)):
    service = LegalOntologyService(LegalOntologyRepository(db))
    return service.create(obj_in)
