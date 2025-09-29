# Placeholder for additional schema logic if needed in the future.
from pydantic import BaseModel


class LegalOntologyBase(BaseModel):
    name: str
    description: str


class LegalOntologyCreate(LegalOntologyBase):
    pass


class LegalOntologyRead(LegalOntologyBase):
    id: int

    class Config:
        from_attributes = True
