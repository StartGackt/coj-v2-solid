
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict

class QAService:
    """
    Service responsible for handling Question Answering logic.
    This encapsulates the ML model and pipeline, adhering to the Single Responsibility Principle.
    """

    def __init__(self):
        """
        Initializes the QA pipeline with a pre-trained model.
        Using a multilingual model suitable for Thai.
        """
        model_name = "distilbert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def answer_question(self, question: str, context: str) -> Dict[str, str | float]:
        """
        Finds an answer to a question within a given context.

        :param question: The question to be answered.
        :param context: The text containing the answer.
        :return: A dictionary containing the answer, score, start, and end positions.
        """
        if not question or not context:
            return {
                "score": 0.0,
                "start": 0,
                "end": 0,
                "answer": "Please provide both a question and a context."
            }

        # The pipeline returns a dictionary with score, start, end, and answer
        result = self.qa_pipeline(question=question, context=context)
        return result
