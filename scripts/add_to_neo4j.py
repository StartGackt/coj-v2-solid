
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.knowledge_graph import KnowledgeGraphService

def main():
    """
    Transforms and loads a legal article's data into the Neo4j knowledge graph.
    """
    # The JSON data for the legal article
    article_data = {
      "article_number": "มาตรา 12",
      "summary": "ห้ามนายจ้างเรียกหลักประกันจากลูกจ้าง ยกเว้นตำแหน่งที่ต้องรับผิดชอบด้านการเงินและต้องปฏิบัติตามประกาศรัฐมนตรี เมื่อต้องคืนหลักประกันต้องดำเนินการภายใน 7 วันพร้อมดอกเบี้ยถ้ามี",
      "obligations": [
        {
          "actor": "นายจ้าง",
          "action": "ต้องไม่เรียกหรือรับหลักประกันจากลูกจ้าง",
          "timeline": None
        },
        {
          "actor": "นายจ้าง",
          "action": "ต้องคืนหลักประกันพร้อมดอกเบี้ยให้ลูกจ้าง",
          "timeline": "ภายใน 7 วันหลังเลิกจ้าง ลาออก หรือสัญญาประกันสิ้นสุด"
        }
      ],
      "exceptions": [
        {
          "description": "อนุญาตให้เรียกหลักประกันเมื่อหน้าที่ลูกจ้างเกี่ยวข้องกับการเงินหรือทรัพย์สินที่เสี่ยงต่อความเสียหาย"
        },
        {
          "description": "รายละเอียดเกี่ยวกับประเภท มูลค่า และการเก็บรักษาหลักประกันเป็นไปตามประกาศรัฐมนตรี"
        }
      ],
      "timelines": [
        "คืนหลักประกันภายใน 7 วันหลังเหตุการณ์สิ้นสุดการจ้างหรือสัญญาประกัน"
      ],
      "compliance_steps": [
        {
          "order": 1,
          "action": "ตรวจสอบและจัดทำบัญชีตำแหน่งที่มีความเสี่ยงด้านการเงินหรือทรัพย์สิน",
          "rationale": "ลดความเสี่ยงในการเรียกหลักประกันเกินจำเป็น"
        },
        {
          "order": 2,
          "action": "จัดทำนโยบายและเอกสารสัญญาที่ระบุหลักเกณฑ์ตามประกาศรัฐมนตรี",
          "rationale": "ให้การเรียกหลักประกันเป็นไปตามข้อกำหนดทางกฎหมาย"
        },
        {
          "order": 3,
          "action": "ตั้งกระบวนการคืนหลักประกันและดอกเบี้ยภายใน 7 วันหลังสิ้นสุดการจ้าง",
          "rationale": "ป้องกันการละเมิดกำหนดเวลาตามกฎหมาย"
        }
      ]
    }

    # Transform the data into the format expected by KnowledgeGraphService
    entities = []
    article_id = article_data["article_number"]

    # Main article node
    article_entity = {
        "id": article_id,
        "label": "LegalArticle",
        "summary": article_data["summary"],
        "relationships": []
    }
    entities.append(article_entity)

    # Obligation nodes and relationships
    for i, ob in enumerate(article_data["obligations"]):
        ob_id = f"{article_id}-obligation-{i}"
        entities.append({
            "id": ob_id,
            "label": "Obligation",
            **ob
        })
        article_entity["relationships"].append({
            "target": ob_id,
            "type": "HAS_OBLIGATION",
            "target_label": "Obligation"
        })

    # Exception nodes and relationships
    for i, ex in enumerate(article_data["exceptions"]):
        ex_id = f"{article_id}-exception-{i}"
        entities.append({
            "id": ex_id,
            "label": "Exception",
            **ex
        })
        article_entity["relationships"].append({
            "target": ex_id,
            "type": "HAS_EXCEPTION",
            "target_label": "Exception"
        })

    # Timeline nodes and relationships
    for i, tl in enumerate(article_data["timelines"]):
        tl_id = f"{article_id}-timeline-{i}"
        entities.append({
            "id": tl_id,
            "label": "Timeline",
            "description": tl
        })
        article_entity["relationships"].append({
            "target": tl_id,
            "type": "HAS_TIMELINE",
            "target_label": "Timeline"
        })

    # Compliance step nodes and relationships
    for i, cs in enumerate(article_data["compliance_steps"]):
        cs_id = f"{article_id}-compliance_step-{i}"
        entities.append({
            "id": cs_id,
            "label": "ComplianceStep",
            **cs
        })
        article_entity["relationships"].append({
            "target": cs_id,
            "type": "HAS_COMPLIANCE_STEP",
            "target_label": "ComplianceStep"
        })

    # Save to Neo4j
    try:
        kg_service = KnowledgeGraphService()
        kg_service.save_entities_and_relationships(entities)
        print("Successfully saved data to Neo4j.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'kg_service' in locals():
            kg_service.close()

if __name__ == "__main__":
    main()
