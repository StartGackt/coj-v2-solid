import sys
import os
import datetime
import json
import argparse
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import chromadb

from app.nlp.dataset import DEFAULT_ARTICLE_FILE, load_legal_articles


DEFAULT_TRIPLET_FILES = [
    os.path.join(os.path.dirname(__file__), "..", "app", "dataset", "semantic_triplets.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the semantic search model with curated triplets"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=DEFAULT_ARTICLE_FILE,
        help="Path to the legal article corpus (default: data1.txt)",
    )
    parser.add_argument(
        "--triplet-file",
        type=str,
        action="append",
        default=None,
        help="JSON file containing triplet seeds (can be passed multiple times)",
    )
    parser.add_argument(
        "--random-negatives",
        type=int,
        default=2,
        help="Additional random negatives to sample per anchor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip before/after semantic search comparison",
    )
    return parser.parse_args()


def load_triplet_files(file_paths: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as seed_file:
                payload = json.load(seed_file)
        except FileNotFoundError:
            print(f"Warning: triplet seed file not found at {path}")
            continue
        except json.JSONDecodeError as exc:
            print(f"Warning: could not parse triplet seed file {path}: {exc}")
            continue

        if not isinstance(payload, list):
            print(f"Warning: seed file {path} must contain a list of entries. Skipping.")
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            anchor = item.get("anchor")
            positive = item.get("positive") or item.get("positive_article")
            positive_text = item.get("positive_text")
            hard_negatives = item.get("hard_negatives", [])
            negative_texts = item.get("negative_texts", [])

            if not anchor or not (positive or positive_text):
                continue

            entries.append(
                {
                    "anchor": anchor.strip(),
                    "positive": positive.strip() if isinstance(positive, str) else None,
                    "positive_text": positive_text.strip() if isinstance(positive_text, str) else None,
                    "hard_negatives": [neg.strip() for neg in hard_negatives if isinstance(neg, str)],
                    "negative_texts": [neg.strip() for neg in negative_texts if isinstance(neg, str)],
                }
            )

    return entries


def create_triplet_dataset(
    articles: dict[str, str],
    seed_entries: list[dict[str, str]],
    random_negatives: int,
    rng: random.Random,
):
    if not seed_entries:
        print("No curated triplet seeds available. Returning empty dataset.")
        return []

    all_article_keys = list(articles.keys())
    if not all_article_keys:
        return []

    train_examples = []
    missing_articles = set()

    for entry in seed_entries:
        anchor = entry["anchor"]
        positive_key = entry.get("positive")
        positive_text = entry.get("positive_text")

        if positive_key:
            positive_text = articles.get(positive_key)
            if not positive_text:
                missing_articles.add(positive_key)
                continue
        elif not positive_text:
            continue

        negative_candidates: list[str] = []
        for key in entry.get("hard_negatives", []):
            if key in articles and key != positive_key:
                negative_candidates.append(key)

        negative_texts = [articles[key] for key in negative_candidates]
        negative_texts.extend(entry.get("negative_texts", []))

        needed_randoms = max(0, random_negatives - len(negative_texts))
        if needed_randoms:
            shuffled = all_article_keys[:]
            rng.shuffle(shuffled)
            for candidate in shuffled:
                if candidate == positive_key or candidate in negative_candidates:
                    continue
                negative_texts.append(articles[candidate])
                needed_randoms -= 1
                if needed_randoms <= 0:
                    break

        for negative_text in negative_texts:
            if not negative_text:
                continue
            train_examples.append(InputExample(texts=[anchor, positive_text, negative_text]))

    if missing_articles:
        missing_list = ", ".join(sorted(missing_articles))
        print(
            f"Warning: skipped {len(missing_articles)} positives missing from source data: {missing_list}"
        )

    rng.shuffle(train_examples)
    return train_examples


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    finetuned_model_dir = f"./models/semantic-search-finetuned-{timestamp}"

    try:
        articles = load_legal_articles(args.data_file)
    except FileNotFoundError as exc:
        print(exc)
        return

    if not articles:
        print("No articles loaded; aborting")
        return

    triplet_files = args.triplet_file or DEFAULT_TRIPLET_FILES
    seed_entries = load_triplet_files(triplet_files)

    # Provide visibility into dataset composition
    print("--- Starting Semantic Search Model Fine-tuning ---")
    print(f"Using {len(seed_entries)} triplet seed entries from {len(triplet_files)} file(s)")

    train_examples = create_triplet_dataset(
        articles=articles,
        seed_entries=seed_entries,
        random_negatives=args.random_negatives,
        rng=rng,
    )
    if not train_examples:
        print("Could not create training examples. Aborting.")
        return

    print(f"Created {len(train_examples)} training triplets.")

    # Load the base model
    model = SentenceTransformer(args.model_name)

    # Create a special dataloader and define the loss function
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(model=model)

    # Fine-tune the model
    print("Fine-tuning the model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=max(1, len(train_examples) // args.batch_size // 10),
    )

    # Save the fine-tuned model
    model.save(finetuned_model_dir)
    print(f"Fine-tuned model saved to {finetuned_model_dir}")

    if not args.skip_compare:
        print("\n--- Comparing search results: Before vs. After Fine-tuning ---")
        query_text = "การเลิกจ้างเพราะลูกจ้างผู้หญิงตั้งท้อง"
        print(f"\nQuery: '{query_text}'")

        # Setup ChromaDB
        client = chromadb.Client()
        ids = list(articles.keys())
        contents = list(articles.values())

        # A) Search with the BASE model
        base_model = SentenceTransformer(args.model_name)
        base_collection = client.get_or_create_collection(
            name="base_model_search", metadata={"hnsw:space": "cosine"}
        )
        base_collection.add(
            embeddings=base_model.encode(contents), documents=contents, ids=ids
        )
        base_results = base_collection.query(
            query_embeddings=base_model.encode([query_text]), n_results=3
        )

        print("\nResults from BASE model:")
        for i, (doc_id, dist) in enumerate(
            zip(base_results["ids"][0], base_results["distances"][0])
        ):
            print(f"{i+1}. {doc_id} (Score: {1 - dist:.4f})")

        # B) Search with the FINE-TUNED model
        finetuned_model = SentenceTransformer(finetuned_model_dir)
        ft_collection = client.get_or_create_collection(
            name="ft_model_search", metadata={"hnsw:space": "cosine"}
        )
        ft_collection.add(
            embeddings=finetuned_model.encode(contents), documents=contents, ids=ids
        )
        ft_results = ft_collection.query(
            query_embeddings=finetuned_model.encode([query_text]), n_results=3
        )

        print("\nResults from FINE-TUNED model:")
        for i, (doc_id, dist) in enumerate(
            zip(ft_results["ids"][0], ft_results["distances"][0])
        ):
            print(f"{i+1}. {doc_id} (Score: {1 - dist:.4f})")


if __name__ == "__main__":
    main()
