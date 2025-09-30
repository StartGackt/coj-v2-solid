from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset


class NLPTrainingService:
    """
    Service class responsible for training NLP models.
    This class adheres to the Single Responsibility Principle by focusing solely on NLP training logic.
    """

    def __init__(self, dataset_loader, model_name="bert-base-cased", num_labels=10):
        """
        Initialize the service with dependencies.

        :param dataset_loader: A callable that loads the dataset.
        :param model_name: Pretrained model name for NER.
        :param num_labels: The number of labels for the classification model.
        """
        self.dataset_loader = dataset_loader
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def train_model(self):
        """
        Load the dataset and train the NLP model.

        :return: Training results or metrics.
        """
        dataset = self.dataset_loader()
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir="./results",  # Directory to save model checkpoints
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()
        return {"status": "success", "message": "Model trained successfully"}

    def _tokenize_function(self, examples):
        """
        Tokenize input examples using the tokenizer.

        :param examples: Input examples to tokenize.
        :return: Tokenized examples.
        """
        tokenized_output = self.tokenizer(examples["text"], padding="max_length", truncation=True)
        if "labels" in examples:
            tokenized_output["labels"] = examples["labels"]
        return tokenized_output
