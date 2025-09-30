from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LegalArticle:
    number: str
    language: str
    text: str


@dataclass(frozen=True)
class ObligationDetail:
    actor: str
    action: str
    timeline: str | None = None


@dataclass(frozen=True)
class ComplianceStepDetail:
    description: str
    rationale: str | None = None


@dataclass(frozen=True)
class LegalArticleAnalysis:
    summary: str
    obligations: list[ObligationDetail]
    exceptions: list[str]
    timelines: list[str]
    compliance_steps: list[ComplianceStepDetail]


class LegalArticleRepositoryProtocol(Protocol):
    def get_article(self, number: str, language: str) -> LegalArticle | None:
        """Retrieve the canonical article text if available."""


class LegalArticleAnalyzerProtocol(ABC):
    @abstractmethod
    def analyze(self, article: LegalArticle) -> LegalArticleAnalysis:
        """Produce a structured analysis for the provided article."""
