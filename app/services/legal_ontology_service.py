# Placeholder for additional service logic if needed in the future.
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.schemas.legal_ontology import LegalOntologyCreate
from typing import List


class LegalOntologyService:
    def __init__(self, repository: LegalOntologyRepository):
        self.repository = repository

    def get_all(self):
        return self.repository.get_all()

    def create(self, obj_in: LegalOntologyCreate):
        return self.repository.create(obj_in.dict())
