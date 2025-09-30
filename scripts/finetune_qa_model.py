
import sys
import os
import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from app.services.qa_finetuning_service import QAFinetuningService
from app.services.qa_service import QAService # To test the new model
from transformers import pipeline

# --- 1. Prepare the Fine-tuning Dataset ---
# For a real use case, you would load a much larger dataset from a file.
# The format is a list of dictionaries, each with 'context', 'question', and 'answers'.
# 'answers' must contain the answer text and the starting character position.

def create_sample_dataset():
    data = [
        {
            "context": "มาตรา 9 \nในกรณีที่นายจ้างไม่คืนหลักประกันที่เป็นเงินตามมาตรา 10 วรรคสอง...ให้นายจ้างเสียดอกเบี้ยให้แก่ลูกจ้างในระหว่างเวลาผิดนัด ร้อยละสิบห้าต่อปี",
            "question": "นายจ้างไม่คืนหลักประกันต้องเสียดอกเบี้ยเท่าไหร่",
            "answers": {"text": ["ร้อยละสิบห้าต่อปี"], "answer_start": [133]},
        },
        {
            "context": "มาตรา 16 ห้ามมิให้นายจ้าง หัวหน้างาน ผู้ควบคุมงาน หรือผู้ตรวจงาน กระทำการล่วงเกิน คุกคาม หรือก่อความเดือดร้อนรำคาญทางเพศต่อลูกจ้าง",
            "question": "ใครบ้างที่ห้ามล่วงละเมิดทางเพศลูกจ้าง",
            "answers": {"text": ["นายจ้าง หัวหน้างาน ผู้ควบคุมงาน หรือผู้ตรวจงาน"], "answer_start": [16]},
        },
        {
            "context": "มาตรา 43 ห้ามมิให้นายจ้างเลิกจ้างลูกจ้างซึ่งเป็นหญิงเพราะเหตุมีครรภ์",
            "question": "นายจ้างสามารถเลิกจ้างลูกจ้างที่ตั้งครรภ์ได้หรือไม่",
            "answers": {"text": ["ห้ามมิให้นายจ้างเลิกจ้างลูกจ้างซึ่งเป็นหญิงเพราะเหตุมีครรภ์"], "answer_start": [9]},
        },
        {
            "context": "มาตรา 44 ห้ามมิให้นายจ้างจ้างเด็กอายุต่ำกว่าสิบห้าปีเป็นลูกจ้าง",
            "question": "จ้างเด็กอายุต่ำกว่ากี่ปีไม่ได้",
            "answers": {"text": ["สิบห้าปี"], "answer_start": [41]},
        },
        {
            "context": "มาตรา 32 ให้ลูกจ้างมีสิทธิลาป่วยได้เท่าที่ป่วยจริง การลาป่วยตั้งแต่สามวันทำงานขึ้นไป นายจ้างอาจให้ลูกจ้างแสดงใบรับรองของแพทย์แผนปัจจุบันชั้นหนึ่ง...",
            "question": "ลาป่วยกี่วันต้องใช้ใบรับรองแพทย์",
            "answers": {"text": ["สามวันทำงานขึ้นไป"], "answer_start": [64]},
        },
        {
            "context": "มาตรา 34 ให้ลูกจ้างมีสิทธิลาเพื่อกิจธุระอันจำเป็นได้ปีละไม่น้อยกว่า สามวันทำงาน",
            "question": "ลูกจ้างมีสิทธิลากิจได้กี่วัน",
            "answers": {"text": ["สามวันทำงาน"], "answer_start": [66]},
        },
    ]
    return Dataset.from_list(data)

# --- 2. Preprocess the Dataset ---
# This function prepares the data for the model, creating start and end token positions.
def preprocess_qa_dataset(dataset, tokenizer):
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return dataset.map(preprocess_function, batched=True)


def main():
    # --- Configuration ---
    base_model = "distilbert-base-multilingual-cased"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    finetuned_model_dir = f"./models/qa-finetuned-{timestamp}"

    # --- 3. Initialize Service and Run Fine-tuning ---
    finetuning_service = QAFinetuningService(
        model_name=base_model,
        tokenizer_name=base_model,
        output_dir=finetuned_model_dir
    )

    raw_dataset = create_sample_dataset()
    tokenized_dataset = preprocess_qa_dataset(raw_dataset, finetuning_service.tokenizer)

    finetuning_service.train_model(tokenized_dataset)

    # --- 4. Test the Fine-tuned Model ---
    print("\n--- Testing the new Fine-tuned Model ---")
    # We re-create the service, but this time pointing to our new, fine-tuned model
    finetuned_qa_service = QAService()
    finetuned_qa_service.tokenizer = finetuning_service.tokenizer
    finetuned_qa_service.model = finetuning_service.model
    finetuned_qa_service.qa_pipeline = pipeline(
        "question-answering", 
        model=finetuned_qa_service.model, 
        tokenizer=finetuned_qa_service.tokenizer
    )

    context = "มาตรา 12 ห้ามมิให้นายจ้างเรียกหรือรับหลักประกันเพื่อการใดๆ จากลูกจ้าง เว้นแต่ลักษณะหรือสภาพของงานที่ทำนั้น ลูกจ้างต้องรับผิดชอบเกี่ยวกับการเงินหรือทรัพย์สินของนายจ้าง"
    question = "ห้ามนายจ้างเรียกหลักประกันจากใคร"

    print(f"Context: {context}")
    print(f"Question: {question}")

    result = finetuned_qa_service.answer_question(question=question, context=context)

    print(f"\nAnswer: '{result['answer']}'")
    print(f"Confidence Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()
