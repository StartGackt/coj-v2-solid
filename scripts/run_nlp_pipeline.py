import sys
import os
import torch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.nlp.nlp_training_service import NLPTrainingService
from app.services.nlp.entity_extraction_service import EntityExtractionService
from app.services.knowledge_graph import KnowledgeGraphService
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.api.dependencies import SessionLocal
from datasets import Dataset

def main():
    """
    Demonstrates the full NLP pipeline: training, extraction, and saving to both
    the relational database and Neo4j.
    """
    # 1. Data Preparation for NER
    # For a real model, you need a much larger and more accurately labeled dataset.
    # Labels: 0: O (Outside), 1: ACTOR (e.g., employer, employee), 2: ITEM (e.g., security deposit)
    text = "ห้ามนายจ้างเรียกหลักประกันจากลูกจ้าง"
    tokens = ["ห้าม", "นายจ้าง", "เรียก", "หลักประกัน", "จาก", "ลูกจ้าง"]
    labels = [0, 1, 0, 2, 0, 1] # Dummy labels for demonstration

    def dataset_loader():
        # The tokenizer will further split words, so we need to align labels.
        # This is a simplified approach. For real use cases, a more robust
        # alignment strategy is needed.
        return Dataset.from_dict({"tokens": [tokens], "ner_tags": [labels]})

    # 2. NLP Model Training
    print("Training NER model...")
    
    # Configure labels
    id2label = {0: "O", 1: "ACTOR", 2: "ITEM"}
    label2id = {"O": 0, "ACTOR": 1, "ITEM": 2}
    
    nlp_training_service = NLPTrainingService(
        dataset_loader, 
        model_name="bert-base-multilingual-cased",
        num_labels=len(id2label)
    )
    nlp_training_service.model.config.id2label = id2label
    nlp_training_service.model.config.label2id = label2id

    # A more complex tokenization function is needed for pre-tokenized input
    def tokenize_and_align_labels(examples):
        tokenized_inputs = nlp_training_service.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    nlp_training_service._tokenize_function = tokenize_and_align_labels
    
    nlp_training_service.train_model()
    print("NER model training completed.")

    # 3. Entity Extraction
    print("\nExtracting entities from the article...")
    extraction_text = "นายจ้างต้องคืนหลักประกันให้ลูกจ้างภายใน 7 วัน"
    entity_extraction_service = EntityExtractionService(
        ner_model=nlp_training_service.model.to("cpu"), 
        tokenizer=nlp_training_service.tokenizer
    )
    extracted_entities = entity_extraction_service.extract_entities(extraction_text)
    print(f"Extracted entities: {extracted_entities}")

    if not extracted_entities:
        print("No entities were extracted, skipping save to databases.")
        return

    # 4. Save to Databases
    # 4.1. Save to Relational Database (PostgreSQL)
    print("\nSaving extracted entities to PostgreSQL...")
    db_session = None
    try:
        db_session = SessionLocal()
        repo = LegalOntologyRepository(db=db_session)
        for entity in extracted_entities:
            print(f"  - Saving entity: {entity['text']} ({entity['entity']})")
            repo.create({"name": entity['text'], "description": entity['entity']})
        print("Successfully saved to PostgreSQL.")
    except Exception as e:
        print(f"An error occurred while saving to PostgreSQL: {e}")
    finally:
        if db_session:
            db_session.close()

    # 4.2. Save to Knowledge Graph (Neo4j)
    print("\nSaving extracted entities to Neo4j...")
    article_id = "มาตรา 12 (demo)"
    entities_to_save_kg = []
    article_entity = {
        "id": article_id,
        "label": "LegalArticle",
        "text": extraction_text,
        "relationships": []
    }
    entities_to_save_kg.append(article_entity)

    for i, entity in enumerate(extracted_entities):
        entity_id = f"{article_id}-entity-{i}"
        entities_to_save_kg.append({
            "id": entity_id,
            "label": entity.get("entity", "Unknown"),
            "text": entity.get("text")
        })
        article_entity["relationships"].append({
            "target": entity_id,
            "type": "CONTAINS_ENTITY",
            "target_label": entity.get("entity", "Unknown")
        })

    kg_service = None
    try:
        kg_service = KnowledgeGraphService()
        kg_service.save_entities_and_relationships(entities_to_save_kg)
        print("Successfully saved extracted entities to Neo4j.")
    except Exception as e:
        print(f"An error occurred while saving to Neo4j: {e}")
    finally:
        if kg_service:
            kg_service.close()

if __name__ == "__main__":
    main()