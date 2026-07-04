import sys
import gc
import argparse
import traceback
from pathlib import Path

project_root = Path("/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/mt_oil_no")
sys.path.insert(0, str(project_root))

import yaml
import torch

from experiments.en_no_expert.a_data_scaling import (
    get_logger,
    load_data_and_evaluator,
    run_one,
    build_data_sizes,
)


def build_all_jobs(cfg, train_ds):
    n = len(train_ds)
    sizes = []
    for s in cfg["experiment"]["train_sizes"]:
        sizes.append(n if s == "full" else int(s))
    seeds = cfg["experiment"]["seeds"]
    return [(size, seed) for size in sizes for seed in seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp1_data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_dir, log_file=f"job_{args.job_id}.log")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
