from app.services.legal_article.factory import build_legal_article_analysis_service


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
