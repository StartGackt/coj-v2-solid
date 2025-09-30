import fastapi 
from app.models.legal_ontology import LegalEntity, LegalRelationship, LegalReference
from app.services.knowledge_graph import KnowledgeGraphService



app = fastapi.FastAPI()

@app.post("/step1")
async def step1(request: Request):
    return {"message": "Hello, World!"}