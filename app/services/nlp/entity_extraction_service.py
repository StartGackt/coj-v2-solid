class EntityExtractionService:
    """
    Service class responsible for extracting entities and relationships from text.
    This adheres to the Single Responsibility Principle by focusing solely on entity extraction logic.
    """

    def __init__(self, ner_model):
        """
        Initialize the service with a pre-trained NER model.

        :param ner_model: A callable pre-trained NER model for entity extraction.
        """
        self.ner_model = ner_model

    def extract_entities(self, text):
        """
        Extract entities and relationships from the given text.

        :param text: The input text to process.
        :return: A list of extracted entities and their relationships.
        """
        predictions = self.ner_model(text)
        entities = self._parse_predictions(predictions)
        return entities

    def _parse_predictions(self, predictions):
        """
        Parse the model predictions to extract entities and relationships.

        :param predictions: Raw predictions from the NER model.
        :return: Parsed entities and relationships.
        """
        # Example parsing logic (to be replaced with actual implementation)
        parsed_results = []
        for prediction in predictions:
            parsed_results.append(
                {
                    "entity": prediction["entity"],
                    "text": prediction["text"],
                    "start": prediction["start"],
                    "end": prediction["end"],
                }
            )
        return parsed_results
