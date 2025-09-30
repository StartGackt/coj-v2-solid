from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class LegalArticleAnalysisRequest(BaseModel):
    article_number: str = Field(
        ..., description="Identifier for the legal article, e.g. 'Section 10'."
    )
    language: Literal["th", "en"] = Field(
        "th", description="Language of the input text."
    )
    text: Optional[str] = Field(
        None,
        description="Full text of the legal article. If omitted, a canonical version will be used if available.",
    )


class LegalObligation(BaseModel):
    actor: str = Field(..., description="Party responsible for the obligation.")
    action: str = Field(..., description="Required action or prohibition.")
    timeline: Optional[str] = Field(
        None, description="Relevant timeframe or deadline, if any."
    )


class LegalException(BaseModel):
    description: str = Field(
        ..., description="Description of the exception or special condition."
    )


class ComplianceStep(BaseModel):
    order: int = Field(
        ..., ge=1, description="Sequential order of the compliance recommendation."
    )
    action: str = Field(..., description="Recommended step to ensure compliance.")
    rationale: Optional[str] = Field(
        None, description="Reasoning or additional context for the step."
    )


class LegalArticleAnalysisResponse(BaseModel):
    article_number: str
    summary: str
    obligations: List[LegalObligation]
    exceptions: List[LegalException]
    timelines: List[str]
    compliance_steps: List[ComplianceStep]


class LegalArticleAnalysisError(BaseModel):
    detail: str
