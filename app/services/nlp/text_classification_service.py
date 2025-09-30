from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch

class TextClassificationService:
    """
    Service class for training text classification models.
    """

    def __init__(self, dataset_loader, model_name="bert-base-multilingual-cased", num_labels=2, output_dir="./results_text_classification"):
        """
        Initialize the service.
        :param dataset_loader: A callable that loads the dataset.
        :param model_name: Pretrained model name.
        :param num_labels: The number of labels for classification.
        :param output_dir: The directory to save training results.
        """
        self.dataset_loader = dataset_loader
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.num_labels = num_labels
        self.output_dir = output_dir

    def train_model(self):
        """
        Load the dataset and train the model.
        """
        dataset = self.dataset_loader()
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
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
        Tokenize input examples.
        """
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def predict(self, text):
        """
        Predict the class of a given text.
        """
        # Move model and inputs to CPU to avoid MPS errors
        self.model.to("cpu")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return {
            "prediction": prediction,
            "label": self.model.config.id2label.get(prediction, "Unknown")
        }
