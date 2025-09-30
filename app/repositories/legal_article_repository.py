from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from app.core.contracts.legal_article import (
    LegalArticle,
    LegalArticleRepositoryProtocol,
)


@dataclass(frozen=True)
class _ArticleRecord:
    number: str
    language: str
    text: str


class InMemoryLegalArticleRepository(LegalArticleRepositoryProtocol):
    """Simple in-memory repository for canonical legal article texts."""

    def __init__(self, records: Dict[Tuple[str, str], _ArticleRecord] | None = None):
        self._records: Dict[Tuple[str, str], _ArticleRecord] = records or {}

    def register(self, record: _ArticleRecord) -> None:
        key = (record.number.lower(), record.language.lower())
        self._records[key] = record

    def get_article(self, number: str, language: str) -> LegalArticle | None:
        key = (number.lower(), language.lower())
        stored = self._records.get(key)
        if not stored:
            return None
        return LegalArticle(
            number=stored.number, language=stored.language, text=stored.text
        )


# Pre-populate repository with Section 10 Thai text
_default_repository = InMemoryLegalArticleRepository()
_default_repository.register(
    _ArticleRecord(
        number="มาตรา 10",
        language="th",
        text=(
            "ภายใต้บังคับมาตรา 51 วรรคหนึ่ง ห้ามมิให้นายจ้างเรียกหรือ รับหลักประกันการทำงานหรือหลักประกันความเสียหายในการทำงาน "
            "ไม่ว่าจะเป็นเงิน ทรัพย์สินอื่น หรือการค้ำประกันด้วยบุคคลจากลูกจ้างเว้นแต่ลักษณะหรือสภาพ ของงานที่ทำนั้นลูกจ้างต้องรับผิดชอบเกี่ยวกับการเงินหรือทรัพย์สินของนายจ้าง "
            "ซึ่งอาจก่อให้เกิดความเสียหายแก่นายจ้างได้ ทั้งนี้ ลักษณะหรือสภาพของงานที่ให้เรียก หรือรับหลักประกันจากลูกจ้าง ตลอดจนประเภทของหลักประกัน จำนวนมูลค่าของ "
            "หลักประกัน และวิธีการเก็บรักษา ให้เป็นไปตามหลักเกณฑ์และวิธีการที่รัฐมนตรี ประกาศกำหนด\n"
            "ในกรณีที่นายจ้างเรียกหรือรับหลักประกัน หรือทำสัญญาประกันกับลูกจ้าง เพื่อชดใช้ความเสียหายที่ลูกจ้างเป็นผู้กระทำ "
            "เมื่อนายจ้างเลิกจ้าง หรือลูกจ้างลาออก หรือสัญญาประกันสิ้นอายุ ให้นายจ้างคืนหลักประกันพร้อมดอกเบี้ย ถ้ามี "
            "แก่ลูกจ้าง ภายในเจ็ดวันนับแต่วันที่นายจ้างเลิกจ้างหรือวันที่ลูกจ้างลาออก หรือวันที่สัญญา ประกันสิ้นอายุ แล้วแต่กรณี"
        ),
    )
)


def get_default_legal_article_repository() -> InMemoryLegalArticleRepository:
    return _default_repository
