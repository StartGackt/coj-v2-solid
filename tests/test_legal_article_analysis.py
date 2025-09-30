from app.services.legal_article.factory import build_legal_article_analysis_service


class StubKnowledgeGraphService:
    def __init__(self) -> None:
        self.payloads = []

    def save_entities_and_relationships(self, entities):
        self.payloads.append(entities)

    def close(self) -> None:  # pragma: no cover - interface parity
        pass


def test_analyze_section_10_from_repository():
    service = build_legal_article_analysis_service()

    analysis = service.analyze_article(
        article_number="มาตรา 10", language="th", text_override=None
    )

    assert "นายจ้างห้ามเรียกหรือรับหลักประกัน" in analysis.summary
    assert any("7 วัน" in timeline for timeline in analysis.timelines)
    assert analysis.obligations[0].actor == "นายจ้าง"
    assert analysis.obligations[1].timeline is not None
    assert len(analysis.compliance_steps) >= 3


def test_analyze_section_10_with_override_text():
    service = build_legal_article_analysis_service()
    custom_text = "มาตรา 10 นายจ้างต้องคืนหลักประกันพร้อมดอกเบี้ยภายใน 7 วัน"

    analysis = service.analyze_article(
        article_number="มาตรา 10", language="th", text_override=custom_text
    )

    assert "คืนหลักประกัน" in analysis.summary
    assert analysis.timelines == [
        "คืนหลักประกันภายใน 7 วันหลังเหตุการณ์สิ้นสุดการจ้างหรือสัญญาประกัน"
    ]


def test_analyze_article_persists_to_knowledge_graph():
    graph_stub = StubKnowledgeGraphService()
    service = build_legal_article_analysis_service(knowledge_graph_service=graph_stub)

    service.analyze_article(
        article_number="มาตรา 10", language="th", text_override=None
    )

    assert graph_stub.payloads, "Knowledge graph should receive persisted entities"
    article_node = graph_stub.payloads[0][0]
    assert article_node["label"] == "LegalArticle"
    assert any(rel["type"] == "HAS_OBLIGATION" for rel in article_node["relationships"])
