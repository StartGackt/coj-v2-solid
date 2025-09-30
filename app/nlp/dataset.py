"""Utility helpers for working with Thai legal datasets."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DEFAULT_ARTICLE_FILE = os.path.join(DATA_DIR, "data1.txt")


_ARTICLE_SPLIT_PATTERN = re.compile(r"\n\s*มาตรา[\s๐-๙\d/]+")
_ARTICLE_TITLE_PATTERN = re.compile(r"มาตรา[\s๐-๙\d/]+")
_CLAUSE_SPLIT_PATTERN = re.compile(r"[\.|\?|!|\n]|(?:\s{2,})")


def load_legal_articles(data_file_path: str | None = None) -> Dict[str, str]:
	"""Load legal articles from the canonical dataset.

	Returns an ordered dictionary mapping article titles (e.g. "มาตรา 43")
	to their corresponding text blocks.
	"""

	file_path = data_file_path or DEFAULT_ARTICLE_FILE
	try:
		with open(file_path, "r", encoding="utf-8") as handle:
			full_text = handle.read()
	except FileNotFoundError:
		raise FileNotFoundError(f"Data file not found at {file_path}") from None

	articles = _ARTICLE_SPLIT_PATTERN.split(full_text)
	article_titles = [title.strip() for title in _ARTICLE_TITLE_PATTERN.findall(full_text)]

	results: "OrderedDict[str, str]" = OrderedDict()
	for title, content in zip(article_titles, articles[1:]):
		cleaned = cleanup_whitespace(content)
		if cleaned:
			results[title] = cleaned
	return results


def cleanup_whitespace(text: str) -> str:
	"""Collapse repeated whitespace and strip leading/trailing spaces."""

	normalized = re.sub(r"\s+", " ", text or "").strip()
	# Remove lingering leading punctuation characters that interfere with snippets
	normalized = normalized.lstrip("-–—•")
	return normalized


def iter_article_clauses(text: str) -> Iterable[str]:
	"""Yield reasonably sized clauses from the article body."""

	normalized = cleanup_whitespace(text)
	if not normalized:
		return []

	raw_clauses = _CLAUSE_SPLIT_PATTERN.split(normalized)
	for clause in raw_clauses:
		candidate = cleanup_whitespace(clause)
		if len(candidate) >= 12:
			yield candidate


def take_leading_phrase(text: str, max_chars: int = 90) -> str:
	"""Return a concise phrase (<= max_chars) suitable for plug-in templates."""

	cleaned = cleanup_whitespace(text)
	if not cleaned:
		return ""

	if len(cleaned) <= max_chars:
		return cleaned

	truncated = cleaned[: max_chars + 1]
	# Avoid cutting mid-word if possible
	last_space = truncated.rfind(" ")
	if last_space > 0:
		truncated = truncated[:last_space]
	return truncated.rstrip(" ,;:()").strip()


def slice_for_keywords(text: str, max_words: int = 12) -> str:
	"""Extract a keyword-focused phrase bounded by max_words."""

	cleaned = cleanup_whitespace(text)
	if not cleaned:
		return ""

	tokens = cleaned.split()
	if len(tokens) <= max_words:
		return cleaned

	snippet = " ".join(tokens[:max_words])
	return snippet.rstrip(" ,;:()").strip()


def collect_candidate_phrases(text: str, max_phrases: int = 6) -> List[str]:
	"""Return a diverse set of keyword phrases from the article body."""

	clauses = list(iter_article_clauses(text))
	phrases: List[str] = []
	for clause in clauses:
		focus = slice_for_keywords(clause)
		if focus and focus not in phrases:
			phrases.append(focus)
		if len(phrases) >= max_phrases:
			break
	return phrases

