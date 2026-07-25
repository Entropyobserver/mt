import sys
import gc
import csv
import json
import logging
import os
import argparse
import statistics
import traceback
from pathlib import Path

project_root = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/mt_oil_no",
    )
)
sys.path.insert(0, str(project_root))

import yaml


def get_logger(output_dir: Path, log_file: str = "experiment.log") -> logging.Logger:
    logger = logging.getLogger(f"exp1_{log_file}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def build_all_jobs(cfg, train_ds):
    n = len(train_ds)
    sizes = []
    for s in cfg["experiment"]["train_sizes"]:
        sizes.append(n if s == "full" else int(s))
    seeds = cfg["experiment"]["seeds"]
    return [(size, seed) for size in sizes for seed in seeds]


def collect_completed_results(output_dir: Path, logger):
    results = []
    skipped = []

    for metrics_file in sorted(output_dir.glob("size_*/seed_*/metrics.json")):
        try:
            with metrics_file.open(encoding="utf-8") as f:
                entry = json.load(f)
        except Exception as exc:
            skipped.append((metrics_file, f"could not read JSON: {exc}"))
            continue

        required = {"data_size", "seed", "test_bleu", "test_chrf"}
        missing = sorted(required - set(entry))
        if missing:
            skipped.append((metrics_file, f"missing keys: {', '.join(missing)}"))
            continue

        results.append(entry)

    if skipped:
        for path, reason in skipped:
            logger.warning(f"Skipping {path}: {reason}")

    if not results:
        raise RuntimeError(f"No completed metrics found under {output_dir}")

    best = max(
        (r for r in results if not r.get("failed")),
        key=lambda r: r.get("test_bleu", float("-inf")),
        default=None,
    )
    best_info = {}
    if best:
        best_info = {
            "size": best["data_size"],
            "seed": best["seed"],
            "test_bleu": best["test_bleu"],
            "model_path": best.get("model_path", ""),
        }

    logger.info(f"Collected {len(results)} completed exp1 runs.")
    return results, best_info


def save_collected_results(results, best_info, output_dir: Path, logger):
    fields = [
        "data_size",
        "seed",
        "val_bleu",
        "val_chrf",
        "val_loss",
        "test_bleu",
        "test_chrf",
        "model_path",
        "failed",
    ]

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if not r.get("failed")]
    grouped = {}
    for entry in valid:
        grouped.setdefault(entry["data_size"], []).append(entry)

    summary_fields = [
        "data_size",
        "test_bleu_mean",
        "test_bleu_std",
        "test_chrf_mean",
        "test_chrf_std",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for size in sorted(grouped):
            rows = grouped[size]
            bleus = [r["test_bleu"] for r in rows]
            chrfs = [r["test_chrf"] for r in rows]
            writer.writerow({
                "data_size": size,
                "test_bleu_mean": round(statistics.mean(bleus), 4),
                "test_bleu_std": round(statistics.stdev(bleus), 4) if len(bleus) > 1 else "",
                "test_chrf_mean": round(statistics.mean(chrfs), 4),
                "test_chrf_std": round(statistics.stdev(chrfs), 4) if len(chrfs) > 1 else "",
            })

    if best_info:
        with (output_dir / "best_model.json").open("w", encoding="utf-8") as f:
            json.dump(best_info, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Collect size_*/seed_*/metrics.json files and write exp1 summary outputs.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp1_data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = "summary.log" if args.summarize else f"job_{args.job_id}.log"
    logger = get_logger(log_dir, log_file=log_file)

    if args.summarize:
        results, best_info = collect_completed_results(output_dir, logger)
        save_collected_results(results, best_info, output_dir, logger)
        return

    if args.job_id is None:
        parser.error("--job_id is required unless --summarize is set")

    from experiments.en_no_expert.a_data_scaling import load_data_and_evaluator, run_one

    train_ds, val_ds, test_ds, evaluator = load_data_and_evaluator(cfg)
    all_jobs = build_all_jobs(cfg, train_ds)

    if args.job_id >= len(all_jobs):
        logger.info(f"job_id={args.job_id} out of range (total={len(all_jobs)}), skipping.")
        return

    size, seed = all_jobs[args.job_id]
    logger.info(f"job_id={args.job_id} size={size} seed={seed} total_jobs={len(all_jobs)}")

    run_dir = output_dir / f"size_{size}" / f"seed_{seed}"
    if (run_dir / "metrics.json").exists():
        logger.info("Already done, skipping.")
        return

    try:
        run_one(size, seed, cfg, train_ds, val_ds, test_ds, evaluator, output_dir, logger)
        logger.info("Done.")
    except Exception as e:
        logger.error(f"FAILED: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
