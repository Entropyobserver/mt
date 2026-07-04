import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TRAIN = ROOT / "data" / "final_splits_npd" / "train.json"
DEFAULT_TEST = ROOT / "data" / "final_splits_npd" / "test.json"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "exp5_final_eval" / "test_predictions.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "contamination_overlap"


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def load_json(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ngram_set(sentences: Iterable[str], n: int) -> set:
    grams = set()
    for sentence in sentences:
        grams.update(ngrams(tokenize(sentence), n))
    return grams


def corpus_overlap(train_sentences: List[str], test_sentences: List[str], n: int) -> Dict[str, float]:
    train_grams = build_ngram_set(train_sentences, n)
    total = 0
    overlap = 0

    for sentence in test_sentences:
        sentence_grams = ngrams(tokenize(sentence), n)
        for gram, count in sentence_grams.items():
            total += count
            if gram in train_grams:
                overlap += count

    ratio = overlap / total if total else 0.0
    return {
        "n": n,
        "test_ngram_count": total,
        "overlap_count": overlap,
        "overlap_ratio": ratio,
    }


def sentence_overlap_ratio(sentence: str, train_grams: set, n: int) -> float:
    sentence_grams = ngrams(tokenize(sentence), n)
    total = sum(sentence_grams.values())
    if total == 0:
        return 0.0
    overlap = sum(count for gram, count in sentence_grams.items() if gram in train_grams)
    return overlap / total


def sentence_overlap_rows(
    train_sentences: List[str],
    test_rows: List[Dict[str, str]],
    field: str,
    n: int,
) -> List[Dict[str, object]]:
    train_grams = build_ngram_set(train_sentences, n)
    rows = []
    for idx, row in enumerate(test_rows):
        ratio = sentence_overlap_ratio(row[field], train_grams, n)
        rows.append(
            {
                "index": idx,
                "overlap_ratio": ratio,
                "source": row.get("source", ""),
                "target": row.get("target", row.get("reference", "")),
            }
        )
    return rows


def exact_duplicate_count(train_sentences: List[str], test_sentences: List[str]) -> int:
    normalized_train = {" ".join(tokenize(sentence)) for sentence in train_sentences}
    return sum(1 for sentence in test_sentences if " ".join(tokenize(sentence)) in normalized_train)


def compute_local_bleu(predictions: List[str], references: List[str], max_order: int = 4) -> Dict[str, float]:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for n in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, n)
            ref_ngrams = ngrams(ref_tokens, n)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(len(pred_tokens) - n + 1, 0)

    precisions = [
        matches_by_order[i] / possible_matches_by_order[i]
        if possible_matches_by_order[i] > 0
        else 0.0
        for i in range(max_order)
    ]

    if min(precisions) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    else:
        geo_mean = 0.0

    ratio = pred_length / ref_length if ref_length > 0 else 0.0
    brevity_penalty = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / ratio) if ratio > 0 else 0.0
    bleu = geo_mean * brevity_penalty

    return {
        "bleu": bleu,
        "bleu_1": precisions[0],
        "bleu_2": precisions[1],
        "bleu_3": precisions[2],
        "bleu_4": precisions[3],
    }


def load_metric_backend():
    try:
        from scripts.evaluation.base_evaluator import BaseEvaluator

        return "scripts.evaluation.BaseEvaluator", BaseEvaluator(use_comet=False)
    except ModuleNotFoundError:
        return "local_bleu_fallback", None


def evaluate_subset(
    prediction_rows: List[Dict[str, str]],
    indices: List[int],
    backend_name: str,
    evaluator,
) -> Dict[str, float]:
    subset = [prediction_rows[i] for i in indices]
    if not subset:
        return {"count": 0}

    predictions = [row["prediction"] for row in subset]
    references = [row["reference"] for row in subset]

    if evaluator is not None:
        metrics = evaluator.evaluate_all(
            sources=[row["source"] for row in subset],
            predictions=predictions,
            references=references,
        )
    else:
        metrics = compute_local_bleu(predictions, references)

    metrics["metric_backend"] = backend_name
    return {"count": len(subset), **metrics}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure train/test n-gram overlap and filtered BLEU for contamination analysis."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sentence-n", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_json(args.train)
    test_rows = load_json(args.test)

    train_sources = [row["source"] for row in train_rows]
    train_targets = [row["target"] for row in train_rows]
    test_sources = [row["source"] for row in test_rows]
    test_targets = [row["target"] for row in test_rows]

    target_overlap = [corpus_overlap(train_targets, test_targets, n) for n in range(1, 5)]
    source_overlap = [corpus_overlap(train_sources, test_sources, n) for n in range(1, 5)]

    sentence_rows = sentence_overlap_rows(
        train_targets,
        test_rows,
        field="target",
        n=args.sentence_n,
    )
    clean_indices = [int(row["index"]) for row in sentence_rows if row["overlap_ratio"] <= args.threshold]
    contaminated_indices = [
        int(row["index"]) for row in sentence_rows if row["overlap_ratio"] > args.threshold
    ]

    top_rows = sorted(sentence_rows, key=lambda row: row["overlap_ratio"], reverse=True)[:25]
    write_csv(args.output_dir / "sentence_overlap_top25.csv", top_rows)

    report = {
        "settings": {
            "train_path": str(args.train),
            "test_path": str(args.test),
            "predictions_path": str(args.predictions),
            "sentence_overlap_n": args.sentence_n,
            "sentence_overlap_threshold": args.threshold,
        },
        "data": {
            "train_sentences": len(train_rows),
            "test_sentences": len(test_rows),
            "exact_source_duplicates": exact_duplicate_count(train_sources, test_sources),
            "exact_target_duplicates": exact_duplicate_count(train_targets, test_targets),
        },
        "target_reference_overlap": target_overlap,
        "source_overlap": source_overlap,
        "sentence_filter": {
            "clean_count": len(clean_indices),
            "contaminated_count": len(contaminated_indices),
            "contaminated_ratio": len(contaminated_indices) / len(test_rows) if test_rows else 0.0,
        },
    }

    if args.predictions.exists():
        prediction_rows = load_json(args.predictions)
        if len(prediction_rows) != len(test_rows):
            raise ValueError(
                f"Prediction rows ({len(prediction_rows)}) do not match test rows ({len(test_rows)})."
            )

        full_indices = list(range(len(prediction_rows)))
        backend_name, evaluator = load_metric_backend()
        full_metrics = evaluate_subset(prediction_rows, full_indices, backend_name, evaluator)
        clean_metrics = evaluate_subset(prediction_rows, clean_indices, backend_name, evaluator)
        contaminated_metrics = evaluate_subset(
            prediction_rows,
            contaminated_indices,
            backend_name,
            evaluator,
        )

        report["filtered_metrics"] = {
            "full": full_metrics,
            "clean": clean_metrics,
            "contaminated": contaminated_metrics,
            "bleu_drop_full_minus_clean": full_metrics.get("bleu", 0.0)
            - clean_metrics.get("bleu", 0.0),
            "chrf_drop_full_minus_clean": full_metrics.get("chrf", 0.0)
            - clean_metrics.get("chrf", 0.0),
        }

    report_path = args.output_dir / "overlap_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved report: {report_path}")
    print("\nTarget/reference overlap, relevant for BLEU inflation:")
    for row in target_overlap:
        print(f"  {row['n']}-gram: {row['overlap_ratio']:.3f} ({row['overlap_ratio'] * 100:.1f}%)")

    print("\nSource/source overlap, relevant for split reuse:")
    for row in source_overlap:
        print(f"  {row['n']}-gram: {row['overlap_ratio']:.3f} ({row['overlap_ratio'] * 100:.1f}%)")

    print(
        f"\nFiltered {len(contaminated_indices)} / {len(test_rows)} test sentences "
        f"with >{args.threshold:.0%} {args.sentence_n}-gram target/reference overlap."
    )
    if "filtered_metrics" in report:
        metrics = report["filtered_metrics"]
        print(
            f"BLEU full={metrics['full']['bleu']:.4f}, "
            f"clean={metrics['clean']['bleu']:.4f}, "
            f"drop={metrics['bleu_drop_full_minus_clean']:.4f}"
        )
        if "chrf" in metrics["full"] and "chrf" in metrics["clean"]:
            print(
                f"chrF full={metrics['full']['chrf']:.2f}, "
                f"clean={metrics['clean']['chrf']:.2f}, "
                f"drop={metrics['chrf_drop_full_minus_clean']:.2f}"
            )
        print(f"Metric backend: {metrics['full']['metric_backend']}")


if __name__ == "__main__":
    main()
