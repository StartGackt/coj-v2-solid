from app.models.legal_ontology import Base
from app.core.config import settings
from sqlalchemy import create_engine


def init_db():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created.")
