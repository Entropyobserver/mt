import sys
import gc
import argparse
import traceback
from pathlib import Path

project_root = Path("/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/mt_oil_no")
sys.path.insert(0, str(project_root))

import yaml
import torch

from experiments.en_no_expert.b_gridsearch import (
    get_logger,
    build_param_grid,
    run_one_config,
)
from scripts.data.data_loader import DataManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))
    train_size = cfg.get("experiment", {}).get("gridsearch", {}).get("train_size", 8000)

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp2_gridsearch"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_dir, log_file=f"job_{args.job_id}.log")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
