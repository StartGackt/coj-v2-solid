from __future__ import annotations

from typing import Optional

from app.core.contracts.legal_article import (
    LegalArticle,
    LegalArticleAnalysis,
    LegalArticleAnalyzerProtocol,
    LegalArticleRepositoryProtocol,
)


class LegalArticleAnalysisService:
    """Coordinates retrieval and analysis of legal articles."""

    def __init__(
        self,
        repository: LegalArticleRepositoryProtocol,
        analyzer: LegalArticleAnalyzerProtocol,
    ) -> None:
        self._repository = repository
        self._analyzer = analyzer

    def analyze_article(
        self,
        article_number: str,
        language: str,
        text_override: Optional[str] = None,
    ) -> LegalArticleAnalysis:
        article = self._resolve_article(article_number, language, text_override)
        return self._analyzer.analyze(article)

    def _resolve_article(
        self, article_number: str, language: str, text_override: Optional[str]
    ) -> LegalArticle:
        if text_override:
            return LegalArticle(
                number=article_number, language=language, text=text_override
            )

        stored_article = self._repository.get_article(article_number, language)
        if stored_article:
            return stored_article

        raise ValueError(
            f"Legal article '{article_number}' in language '{language}' is not available and no text override was provided."
        )
