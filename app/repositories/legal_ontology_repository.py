# Placeholder for additional repository logic if needed in the future.
from app.models.legal_ontology import LegalOntology
from typing import List


class LegalOntologyRepository:
    def __init__(self, db):
        self.db = db

    def get_all(self) -> List[LegalOntology]:
        return self.db.query(LegalOntology).all()

    def create(self, obj_in: dict) -> LegalOntology:
        obj = LegalOntology(**obj_in)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj
