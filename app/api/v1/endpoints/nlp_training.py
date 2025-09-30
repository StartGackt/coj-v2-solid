from fastapi import APIRouter, Depends
from app.services.nlp.nlp_training_service import NLPTrainingService
from datasets import Dataset

router = APIRouter()


# Dependency injection for the NLPTrainingService
def get_nlp_training_service():
    def dataset_loader():
        # Example dataset for legal Ontology/Schema
        data = {
            "text": [
                "มาตรา 10 ห้ามมิให้นายจ้างเรียกหลักประกัน",
                "มาตรา 51 นายจ้างต้องคืนหลักประกัน",
                "มาตรา 20 ลูกจ้างต้องปฏิบัติตามคำสั่งนายจ้าง",
            ]
        }
        return Dataset.from_dict(data)

    return NLPTrainingService(dataset_loader)


@router.post(
    "/train-nlp",
    tags=["NLP Training"],
    summary="Train NLP Model",
    description="This endpoint triggers the training of an NLP model using a predefined dataset and model configuration.",
)
def train_nlp(service: NLPTrainingService = Depends(get_nlp_training_service)):
    """
    Train an NLP model.

    This endpoint allows users to train an NLP model, such as a Named Entity Recognition (NER) model, using a predefined dataset.

    - **Input**: None (uses a predefined dataset internally).
    - **Output**: Training results or metrics.
    """
    results = service.train_model()
    return {"message": "Training completed", "results": results}
