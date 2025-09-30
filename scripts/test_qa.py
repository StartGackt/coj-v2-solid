
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.qa_service import QAService

def main():
    """
    A simple script to test the QAService.
    """
    print("Initializing QA Service... (This may take a moment to download the model)")
    qa_service = QAService()
    print("Service Initialized.")

    # --- Example 1 ---
    print("\n--- Example 1: Legal Context ---")
    context = "มาตรา 12 ห้ามมิให้นายจ้างเรียกหรือรับหลักประกันเพื่อการใดๆ จากลูกจ้าง เว้นแต่ลักษณะหรือสภาพของงานที่ทำนั้น ลูกจ้างต้องรับผิดชอบเกี่ยวกับการเงินหรือทรัพย์สินของนายจ้าง"
    question = "ห้ามนายจ้างเรียกหลักประกันจากใคร"

    print(f"Context: {context}")
    print(f"Question: {question}")

    result = qa_service.answer_question(question=question, context=context)

    print(f"\nAnswer: '{result['answer']}'")
    print(f"Confidence Score: {result['score']:.4f}")

    # --- Example 2 ---
    print("\n--- Example 2: General Context ---")
    context = "ประธานาธิบดีคนปัจจุบันของสหรัฐอเมริกาคือโจ ไบเดน เขาเข้ารับตำแหน่งในปี 2021"
    question = "ใครคือประธานาธิบดีคนปัจจุบัน"

    print(f"Context: {context}")
    print(f"Question: {question}")

    result = qa_service.answer_question(question=question, context=context)

    print(f"\nAnswer: '{result['answer']}'")
    print(f"Confidence Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()
