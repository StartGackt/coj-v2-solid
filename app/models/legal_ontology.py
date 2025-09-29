from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

# For Alembic migrations
from sqlalchemy.orm import DeclarativeMeta

metadata = Base.metadata


class LegalOntology(Base):
    __tablename__ = "legal_ontology"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
