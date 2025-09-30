
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

class QAFinetuningService:
    """
    Service class for fine-tuning Question Answering models.
    Adheres to SRP by focusing solely on the training/fine-tuning logic.
    """

    def __init__(self, model_name: str, tokenizer_name: str, output_dir: str):
        """
        Initializes the service.

        :param model_name: The base model to be fine-tuned.
        :param tokenizer_name: The tokenizer to use.
        :param output_dir: The directory to save the fine-tuned model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.output_dir = output_dir

    def train_model(self, train_dataset):
        """
        Fine-tunes the QA model on the provided dataset.

        :param train_dataset: The processed and tokenized training dataset.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=8, # Smaller batch size for QA tasks
            num_train_epochs=3,
            learning_rate=3e-5,
            weight_decay=0.01,
            # No evaluation for this simple demo
            # evaluation_strategy="epoch", 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset # Optionally add an evaluation set
        )

        print("Starting model fine-tuning...")
        trainer.train()
        print("Fine-tuning completed.")

        # Save the final model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Fine-tuned model saved to {self.output_dir}")

        return {"status": "success", "output_dir": self.output_dir}
