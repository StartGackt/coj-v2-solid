from pydantic import BaseModel


class LegalOntologyBase(BaseModel):
    name: str
    description: str


class LegalOntologyCreate(LegalOntologyBase):
    pass


class LegalOntologyRead(LegalOntologyBase):
    id: int

    class Config:
        orm_mode = True
