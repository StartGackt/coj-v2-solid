import fastapi 
from app.models.legal_ontology import LegalEntity, LegalRelationship, LegalReference
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.legal_document_processor import LegalDocumentProcessor
from app.services.legal_reference_processor import LegalReferenceProcessor
from app.services.legal_relationship_processor import LegalRelationshipProcessor
from app.services.legal_entity_processor import LegalEntityProcessor
from app.services.legal_document_processor import LegalDocumentProcessor
from app.services.legal_reference_processor import LegalReferenceProcessor
from app.services.legal_relationship_processor import LegalRelationshipProcessor
from app.services.legal_entity_processor import LegalEntityProcessor


app = fastapi.FastAPI()

@app.post("/step1")
async def step1(request: Request):
    return {"message": "Hello, World!"}