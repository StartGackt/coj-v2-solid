# Placeholder for additional app logic if needed in the future.
def main():
    import uvicorn
    from fastapi import FastAPI
    from app.api.v1.endpoints import legal_ontology, nlp_training

    app = FastAPI()
    app.include_router(legal_ontology.router, prefix="/api/v1")
    app.include_router(nlp_training.router, prefix="/api/v1", tags=["NLP Training"])
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
