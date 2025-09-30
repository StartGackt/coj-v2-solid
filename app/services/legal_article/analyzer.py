from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.contracts.legal_article import (
    ComplianceStepDetail,
    LegalArticle,
    LegalArticleAnalysis,
    LegalArticleAnalyzerProtocol,
    ObligationDetail,
)


@dataclass(frozen=True)
class AnalysisRule:
    keyword: str
    summary_point: str


class HeuristicLegalArticleAnalyzer(LegalArticleAnalyzerProtocol):
    """A simple rule-based analyzer leveraging keywords to craft structured insights."""

    def __init__(self, summary_rules: List[AnalysisRule] | None = None):
        self._summary_rules = summary_rules or []

    def analyze(self, article: LegalArticle) -> LegalArticleAnalysis:
        normalized_text = article.text.replace("\n", " ").strip()
        summary_sentences = self._build_summary(normalized_text)

        obligations = [
            ObligationDetail(
                actor="นายจ้าง",
                action="ต้องไม่เรียกหรือรับหลักประกันจากลูกจ้าง",
                timeline=None,
            ),
            ObligationDetail(
                actor="นายจ้าง",
                action="ต้องคืนหลักประกันพร้อมดอกเบี้ยให้ลูกจ้าง",
                timeline="ภายใน 7 วันหลังเลิกจ้าง ลาออก หรือสัญญาประกันสิ้นสุด",
            ),
        ]
        exceptions = [
            "อนุญาตให้เรียกหลักประกันเมื่อหน้าที่ลูกจ้างเกี่ยวข้องกับการเงินหรือทรัพย์สินที่เสี่ยงต่อความเสียหาย",
            "รายละเอียดเกี่ยวกับประเภท มูลค่า และการเก็บรักษาหลักประกันเป็นไปตามประกาศรัฐมนตรี",
        ]
        timelines = [
            "คืนหลักประกันภายใน 7 วันหลังเหตุการณ์สิ้นสุดการจ้างหรือสัญญาประกัน",
        ]
        compliance_steps = [
            ComplianceStepDetail(
                description="ตรวจสอบและจัดทำบัญชีตำแหน่งที่มีความเสี่ยงด้านการเงินหรือทรัพย์สิน",
                rationale="ลดความเสี่ยงในการเรียกหลักประกันเกินจำเป็น",
            ),
            ComplianceStepDetail(
                description="จัดทำนโยบายและเอกสารสัญญาที่ระบุหลักเกณฑ์ตามประกาศรัฐมนตรี",
                rationale="ให้การเรียกหลักประกันเป็นไปตามข้อกำหนดทางกฎหมาย",
            ),
            ComplianceStepDetail(
                description="ตั้งกระบวนการคืนหลักประกันและดอกเบี้ยภายใน 7 วันหลังสิ้นสุดการจ้าง",
                rationale="ป้องกันการละเมิดกำหนดเวลาตามกฎหมาย",
            ),
        ]

        return LegalArticleAnalysis(
            summary=" ".join(summary_sentences),
            obligations=obligations,
            exceptions=exceptions,
            timelines=timelines,
            compliance_steps=compliance_steps,
        )

    def _build_summary(self, text: str) -> List[str]:
        matches = [
            rule.summary_point for rule in self._summary_rules if rule.keyword in text
        ]
        if not matches:
            matches.append(
                "ห้ามนายจ้างเรียกหลักประกันจากลูกจ้าง ยกเว้นตำแหน่งที่ต้องรับผิดชอบด้านการเงินและต้องปฏิบัติตามประกาศรัฐมนตรี"
            )
        if not any("คืนหลักประกัน" in point for point in matches):
            matches.append("เมื่อต้องคืนหลักประกันต้องดำเนินการภายใน 7 วันพร้อมดอกเบี้ยถ้ามี")
        return matches
