from __future__ import annotations

from app.repositories.legal_article_repository import (
    get_default_legal_article_repository,
)
from app.services.legal_article.analyzer import (
    AnalysisRule,
    HeuristicLegalArticleAnalyzer,
)
from app.services.legal_article.service import LegalArticleAnalysisService
from app.services.knowledge_graph import KnowledgeGraphService


def build_legal_article_analysis_service(
    *,
    knowledge_graph_service: KnowledgeGraphService | None = None,
) -> LegalArticleAnalysisService:
    repository = get_default_legal_article_repository()
    analyzer = HeuristicLegalArticleAnalyzer(
        summary_rules=[
            AnalysisRule(
                keyword="ห้ามมิให้นายจ้างเรียกหรือ รับหลักประกัน",
                summary_point="นายจ้างห้ามเรียกหรือรับหลักประกันจากลูกจ้างโดยทั่วไป",
            ),
            AnalysisRule(
                keyword="ลูกจ้างต้องรับผิดชอบเกี่ยวกับการเงินหรือทรัพย์สินของนายจ้าง",
                summary_point="อนุญาตให้เรียกหลักประกันเมื่อหน้าที่เกี่ยวข้องกับการเงินหรือทรัพย์สินเสี่ยง",
            ),
            AnalysisRule(
                keyword="คืนหลักประกันพร้อมดอกเบี้ย",
                summary_point="ต้องคืนหลักประกันพร้อมดอกเบี้ยเมื่อสิ้นสุดการจ้าง",
            ),
        ]
    )
    return LegalArticleAnalysisService(
        repository=repository,
        analyzer=analyzer,
        knowledge_graph=knowledge_graph_service,
    )
