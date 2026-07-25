import sys
import gc
import csv
import json
import logging
import os
import argparse
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
    logger = logging.getLogger(f"exp2_{log_file}")
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


def collect_completed_results(output_dir: Path, logger):
    results = []
    skipped = []

    for metrics_file in sorted(output_dir.glob("r*_a*_d*/metrics.json")):
        try:
            with metrics_file.open(encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as exc:
            skipped.append((metrics_file, f"could not read JSON: {exc}"))
            continue

        config = metrics.get("config", {})
        val_metrics = metrics.get("val_metrics", {})
        required_config = {"r", "alpha", "dropout"}
        required_metrics = {"bleu", "chrf", "loss"}
        missing = sorted(required_config - set(config)) + sorted(required_metrics - set(val_metrics))
        if missing:
            skipped.append((metrics_file, f"missing keys: {', '.join(missing)}"))
            continue

        run_dir = metrics_file.parent
        results.append({
            "r": config["r"],
            "alpha": config["alpha"],
            "dropout": config["dropout"],
            "val_bleu": val_metrics["bleu"],
            "val_chrf": val_metrics["chrf"],
            "val_loss": val_metrics["loss"],
            "model_path": str(run_dir / "training" / "final_model"),
            "training_time_seconds": metrics.get("training_time", 0),
        })

    if skipped:
        for path, reason in skipped:
            logger.warning(f"Skipping {path}: {reason}")

    if not results:
        raise RuntimeError(f"No completed grid-search metrics found under {output_dir}")

    logger.info(f"Collected {len(results)} completed exp2 configs.")
    return results


def save_collected_results(results, output_dir: Path, logger):
    fields = [
        "r",
        "alpha",
        "dropout",
        "val_bleu",
        "val_chrf",
        "val_loss",
        "model_path",
        "training_time_seconds",
        "failed",
    ]

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if not r.get("failed")]
    if not valid:
        logger.warning("No successful grid-search configs to summarize.")
        return

    top5 = sorted(valid, key=lambda r: r["val_bleu"], reverse=True)[:5]
    logger.info("Top 5 configurations by validation BLEU:")
    for idx, row in enumerate(top5, 1):
        logger.info(
            f"  {idx}. r={row['r']}, alpha={row['alpha']}, dropout={row['dropout']}  "
            f"val BLEU={row['val_bleu']:.4f}"
        )

    best = top5[0]
    with (output_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "r": best["r"],
                "alpha": best["alpha"],
                "dropout": best["dropout"],
                "val_bleu": best["val_bleu"],
                "model_path": best["model_path"],
            },
            f,
            indent=2,
        )

    logger.info(
        f"Best by validation BLEU: r={best['r']}, alpha={best['alpha']}, "
        f"dropout={best['dropout']}, val BLEU={best['val_bleu']:.4f}"
    )
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Collect r*_a*_d*/metrics.json files and write grid-search summary outputs.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))
    train_size = cfg.get("experiment", {}).get("gridsearch", {}).get("train_size", 8000)

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp2_gridsearch"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = "summary.log" if args.summarize else f"job_{args.job_id}.log"
    logger = get_logger(log_dir, log_file=log_file)

    if args.summarize:
        results = collect_completed_results(output_dir, logger)
        save_collected_results(results, output_dir, logger)
        return

    if args.job_id is None:
        parser.error("--job_id is required unless --summarize is set")

    from experiments.en_no_expert.b_gridsearch import build_param_grid, run_one_config
    from scripts.data.data_loader import DataManager

    all_configs = build_param_grid(cfg)

    if args.job_id >= len(all_configs):
        logger.error(f"job_id={args.job_id} out of range (total={len(all_configs)})")
        sys.exit(1)

    r, alpha, dropout = all_configs[args.job_id]
    logger.info(f"job_id={args.job_id} r={r} alpha={alpha} dropout={dropout} train_size={train_size}")

    config_name = f"r{r}_a{alpha}_d{dropout}"
    run_dir = output_dir / config_name
    if (run_dir / "metrics.json").exists():
        logger.info("Already done, skipping.")
        return

    data_manager = DataManager(cfg)
    train_ds, val_ds, _ = data_manager.load_splits()

    train_ds = train_ds.subset(train_size, seed=cfg["project"]["seed"])
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]

    try:
        run_one_config(r, alpha, dropout, cfg, train_data, val_data, output_dir, logger)
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
