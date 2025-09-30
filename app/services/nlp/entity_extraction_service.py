import torch

class EntityExtractionService:
    """
    Service class responsible for extracting entities and relationships from text.
    This adheres to the Single Responsibility Principle by focusing solely on entity extraction logic.
    """

    def __init__(self, ner_model, tokenizer):
        """
        Initialize the service with a pre-trained NER model and tokenizer.

        :param ner_model: A callable pre-trained NER model for entity extraction.
        :param tokenizer: The tokenizer corresponding to the NER model.
        """
        self.ner_model = ner_model
        self.tokenizer = tokenizer

    def extract_entities(self, text):
        """
        Extract entities and relationships from the given text.

        :param text: The input text to process.
        :return: A list of extracted entities and their relationships.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        
        entities = self._parse_predictions(predictions, inputs)
        return entities

    def _parse_predictions(self, predictions, inputs):
        """
        Parse the model predictions to extract entities and relationships,
        grouping subword tokens into complete entities.

        :param predictions: Raw predictions from the NER model.
        :param inputs: The tokenized inputs.
        :return: A list of dictionaries, where each dictionary represents an entity.
        """
        predictions = predictions[0].tolist()
        input_ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        entities = []
        current_entity = None

        for token, pred_id in zip(tokens, predictions):
            if token in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                continue

            pred_label = self.ner_model.config.id2label[pred_id]

            if pred_label != 'O': # Assuming 'O' is the "Outside" tag
                if current_entity and current_entity["entity"] == pred_label:
                    # Continue current entity
                    current_entity["text"] += token.replace("##", "")
                else:
                    # Start a new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {"entity": pred_label, "text": token.replace("##", "")}
            else:
                # End of an entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if it exists
        if current_entity:
            entities.append(current_entity)

        return entities
