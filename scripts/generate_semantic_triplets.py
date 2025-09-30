"""Generate an expanded semantic search triplet dataset for Thai legal articles."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sentence_transformers import SentenceTransformer

from app.nlp.dataset import (
    DEFAULT_ARTICLE_FILE,
    cleanup_whitespace,
    collect_candidate_phrases,
    load_legal_articles,
    take_leading_phrase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic triplets")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(DEFAULT_ARTICLE_FILE),
        help="Path to the legal article corpus (default: data1.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/dataset/semantic_triplets_generated.json"),
        help="Where to store the generated triplets",
    )
    parser.add_argument(
        "--per-article",
        type=int,
        default=45,
        help="Number of anchors to produce per legal article",
    )
    parser.add_argument(
        "--negatives-per-anchor",
        type=int,
        default=10,
        help="Number of negatives to attach to each anchor",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used for embeddings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def build_context_variants(phrases: Sequence[str]) -> List[str]:
    variants: List[str] = []
    for phrase in phrases:
        base = cleanup_whitespace(phrase)
        if base and base not in variants:
            variants.append(base)
    return variants


def build_anchor_templates() -> List[str]:
    return [
        "มาตราใดกล่าวถึง {context}",
        "ถ้าเกิด {context} กฎหมายแรงงานกำหนดไว้อย่างไร",
        "ข้อกำหนดเรื่อง {context} อยู่ในมาตราใดของ พ.ร.บ.คุ้มครองแรงงาน",
        "ประเด็น {context} ลูกจ้างและนายจ้างต้องปฏิบัติตามกฎหมายข้อไหน",
        "กฎหมายแรงงานคุ้มครองกรณี {context} อย่างไร",
        "กรณี {context} นายจ้างควรอ้างถึงบทกฎหมายใด",
        "สิทธิหน้าที่เมื่อ {context} ถูกระบุไว้ในมาตราใด",
        "หากสถานการณ์คือ {context} ต้องดูมาตราใด",
        "คำถาม: {context} อยู่ในบทบัญญัติใด",
        "เมื่อ {context} ผู้เกี่ยวข้องควรปฏิบัติตามมาตราใด",
        "ถ้า {context} เกิดขึ้น ใครมีภาระหน้าที่ตามกฎหมายและอยู่ในมาตราใด",
        "อธิบาย {context} ต้องดูบทบัญญัติใดของกฎหมายแรงงาน",
    ]


def generate_anchor_prompts(
    article_text: str,
    per_article: int,
    rng: random.Random,
) -> List[str]:
    phrases = collect_candidate_phrases(article_text, max_phrases=8)
    if not phrases:
        fallback = take_leading_phrase(article_text)
        if fallback:
            phrases = [fallback]

    contexts = build_context_variants(phrases)
    if not contexts:
        contexts = [take_leading_phrase(article_text) or "ประเด็นกฎหมายนี้"]

    templates = build_anchor_templates()
    anchors: List[str] = []
    for context in contexts:
        for template in templates:
            anchors.append(template.format(context=context))

    # Deduplicate while preserving order
    seen = set()
    unique_anchors: List[str] = []
    for anchor in anchors:
        cleaned = cleanup_whitespace(anchor)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_anchors.append(cleaned)

    rng.shuffle(unique_anchors)

    if len(unique_anchors) < per_article:
        # Repeat with slight emphasis to reach the target count
        augmented = []
        suffixes = [
            "โปรดระบุมาตราในกรณี: {anchor}",
            "อ้างถึงมาตราใด: {anchor}",
            "เพื่อใช้ในการอบรม บุคลากรควรรู้ว่า {anchor}",
        ]
        for anchor in unique_anchors:
            augmented.append(anchor)
            for suffix in suffixes:
                augmented.append(suffix.format(anchor=anchor))
        unique_anchors = augmented
        rng.shuffle(unique_anchors)

    return unique_anchors[:per_article]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    similarity = np.matmul(normalized, normalized.T)
    return similarity


def pick_negatives(
    similarity: np.ndarray,
    index: int,
    article_titles: Sequence[str],
    negatives_per_anchor: int,
    rng: random.Random,
) -> List[str]:
    row = similarity[index]
    candidate_indices = np.argsort(row)[::-1]
    hard_pool = [idx for idx in candidate_indices if idx != index][: max(negatives_per_anchor * 3, 20)]

    if not hard_pool:
        hard_pool = [idx for idx in range(len(article_titles)) if idx != index]

    negatives: List[str] = []
    for idx in hard_pool[: negatives_per_anchor // 2]:
        negatives.append(article_titles[idx])

    while len(negatives) < negatives_per_anchor and hard_pool:
        choice = rng.choice(hard_pool)
        candidate_title = article_titles[choice]
        if candidate_title not in negatives and candidate_title != article_titles[index]:
            negatives.append(candidate_title)

    # Fallback random sampling if still short
    while len(negatives) < negatives_per_anchor:
        choice = rng.randint(0, len(article_titles) - 1)
        candidate_title = article_titles[choice]
        if candidate_title not in negatives and candidate_title != article_titles[index]:
            negatives.append(candidate_title)

    return negatives


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    articles = load_legal_articles(str(args.data_file))
    article_titles = list(articles.keys())
    article_texts = list(articles.values())

    print(f"Loaded {len(article_titles)} articles from {args.data_file}")

    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(article_texts, show_progress_bar=True)
    similarity = compute_similarity_matrix(np.asarray(embeddings))

    triplets = []
    global_seen_anchors = set()

    for idx, (title, body) in enumerate(articles.items()):
        anchors = generate_anchor_prompts(body, args.per_article, rng)
        negatives = pick_negatives(
            similarity,
            index=idx,
            article_titles=article_titles,
            negatives_per_anchor=args.negatives_per_anchor,
            rng=rng,
        )

        for anchor in anchors:
            if anchor in global_seen_anchors:
                continue
            triplets.append(
                {
                    "anchor": anchor,
                    "positive": title,
                    "hard_negatives": negatives,
                }
            )
            global_seen_anchors.add(anchor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(triplets, handle, ensure_ascii=False, indent=2)

    print(
        f"Generated {len(triplets)} triplets across {len(article_titles)} articles and saved to {args.output}"
    )


if __name__ == "__main__":
    main()
