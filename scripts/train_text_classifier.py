
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.nlp.text_classification_service import TextClassificationService
from datasets import Dataset

import datetime

# ... (rest of the imports)

def main():
    """
    Trains and demonstrates a text classification model for identifying legal text.
    """
    # --- Configuration ---
    # You can change this name for each experiment
    experiment_name = "legal_vs_non_legal_v1"
    # ---------------------

    # Create a descriptive, timestamped output directory for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./results/{experiment_name}/{timestamp}"
    print(f"Training results will be saved to: {output_dir}")

    # 1. Data Preparation
    # For a real model, you need a large and balanced dataset of legal and non-legal texts.
    legal_text = "ห้ามนายจ้างเรียกหลักประกันจากลูกจ้าง ยกเว้นตำแหน่งที่ต้องรับผิดชอบด้านการเงินและต้องปฏิบัติตามประกาศรัฐมนตรี"
    non_legal_text = "วันนี้อากาศดีมาก ท้องฟ้าแจ่มใส เหมาะแก่การไปเที่ยวพักผ่อน"

    # Labels: 1 for legal, 0 for non-legal
    def dataset_loader():
        data = {
            "text": [legal_text, non_legal_text],
            "label": [1, 0]
        }
        return Dataset.from_dict(data)

    # 2. Model Training
    print("Training text classification model...")
    # I'm using a multilingual model which is a good starting point for Thai.
    classification_service = TextClassificationService(
        dataset_loader, 
        model_name="bert-base-multilingual-cased", 
        num_labels=2,
        output_dir=output_dir
    )
    
    # Configure labels
    classification_service.model.config.id2label = {0: "Non-Legal", 1: "Legal"}
    classification_service.model.config.label2id = {"Non-Legal": 0, "Legal": 1}

    classification_service.train_model()
    print("Model training completed.")

    # 3. Prediction
    print("\n--- Testing the trained model ---")
    
    test_legal = "ลูกจ้างมีสิทธิได้รับค่าชดเชยหากถูกเลิกจ้างโดยไม่มีความผิด"
    prediction = classification_service.predict(test_legal)
    print(f"Text: \"{test_legal}\"")
    print(f"Prediction: {prediction['label']}\n")

    test_non_legal = "ร้านอาหารนี้อร่อยมาก โดยเฉพาะเมนูผัดไทยกุ้งสด"
    prediction = classification_service.predict(test_non_legal)
    print(f"Text: \"{test_non_legal}\"")
    print(f"Prediction: {prediction['label']}")

if __name__ == "__main__":
    main()
