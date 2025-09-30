from __future__ import annotations

from typing import List

from app.core.contracts.legal_article import LegalArticleAnalysis
from app.schemas.legal_article import (
    ComplianceStep,
    LegalArticleAnalysisResponse,
    LegalException,
    LegalObligation,
)


def map_analysis_to_response(
    article_number: str, analysis: LegalArticleAnalysis
) -> LegalArticleAnalysisResponse:
    obligations = [
        LegalObligation(
            actor=item.actor,
            action=item.action,
            timeline=item.timeline,
        )
        for item in analysis.obligations
    ]
    exceptions = [LegalException(description=item) for item in analysis.exceptions]
    compliance_steps = [
        ComplianceStep(
            order=index + 1,
            action=item.description,
            rationale=item.rationale,
        )
        for index, item in enumerate(analysis.compliance_steps)
    ]

    return LegalArticleAnalysisResponse(
        article_number=article_number,
        summary=analysis.summary,
        obligations=obligations,
        exceptions=exceptions,
        timelines=analysis.timelines,
        compliance_steps=compliance_steps,
    )
