"""
LoRA vs Full Fine-tuning (EN-NO)
Research questions:
- Does full fine-tuning outperform LoRA on this low-resource translation task?
- How do the two methods compare across different training data sizes?
- What are the training time trade-offs between the two methods?
Fairness guarantees:
- Same backbone: NLLB-200-distilled-600M
- Same training data, same subset sampling logic (same seeds)
- Same epochs, batch size, gradient accumulation, warmup, eval_steps
- Same test set evaluation
- Learning rates differ by design (5e-4 for LoRA, 5e-5 for full FT) —
  this is standard practice, not an advantage for either method:
  LoRA uses a larger lr because only ~2M parameters are updated,
  full FT uses a smaller lr to prevent divergence across all 600M parameters
- Full FT uses fp32 instead of fp16 for training stability —
  not a performance advantage, just a numerical requirement
LoRA configuration note:
  Both methods use the default LoRA config (r=16, alpha=32, dropout=0.1),
  NOT the optimised config from exp2 (r=8, alpha=64, dropout=0.0).
  This is intentional: we compare methods under equal conditions,
  without giving LoRA the benefit of hyperparameter tuning.
Usage:
    python e_lora_vs_ft.py --method lora
    python e_lora_vs_ft.py --method ft
    python e_lora_vs_ft.py --method all   (runs both sequentially)
"""
import os
import sys
import gc
import json
import time
import logging
import traceback
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

HF_CACHE_DIR = "/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/hf_cache"
os.environ.update({
    "HF_HOME": HF_CACHE_DIR,
    "HF_DATASETS_CACHE": HF_CACHE_DIR,
    "TORCH_HOME": HF_CACHE_DIR,
})

from scripts.model.lora_trainer import LoRATrainer
from scripts.model.full_trainer import FullTrainer
from scripts.data.data_loader import DataManager
from scripts.evaluation.base_evaluator import BaseEvaluator

# learning rates differ by design — standard practice for each method
LR = {
    "lora": 5e-4,
    "ft": 5e-5,
}


def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp6_method_comparison")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def build_data_sizes(train_ds) -> list:
    # mirror exp1 but skip 100/500 — too small for method comparison
    candidates = [1000, 2000, 4000, 6000, 8000, 10000]
    n = len(train_ds)
    sizes = sorted(set(s for s in candidates if s < n) | {n})
    return sizes


def build_train_config(cfg, method, seed, run_dir) -> dict:
    return {
        "output_dir": str(run_dir / "training"),
        "seed":  seed,
        "epochs": cfg["training"]["epochs"],
        "batch_size":  cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": LR[method],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16":   method == "lora",  # ft uses fp32 for stability
        "max_length": cfg["model"].get("max_length", 128),
        "generation_max_length": cfg["generation"].get("max_length", cfg["model"].get("max_length", 128)),
        "generation_num_beams": cfg["generation"]["num_beams"],
        "save_total_limit": 1,
        "save_final_model": method == "lora",  # ft model deleted after eval
    }


def get_trainer(cfg, method):
    if method == "lora":
        return LoRATrainer(
            model_name=cfg["model"]["pretrained"],
            src_lang=cfg["model"]["src_lang"],
            tgt_lang=cfg["model"]["tgt_lang"],
        )
    return FullTrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )


def run_one(method, size, seed, cfg, train_ds, val_ds, test_ds, evaluator, output_dir, logger):
    size_label = "full" if size == len(train_ds) else str(size)
    run_dir = output_dir / method / f"size{size_label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(cfg, method)
    train_config = build_train_config(cfg, method, seed, run_dir)

    try:
        train_subset = train_ds.subset(size, seed=seed)
        train_data = [s.to_dict() for s in train_subset.samples]
        val_data = [s.to_dict() for s in val_ds.samples]

        start_time = time.time()
        train_result = trainer.train(train_data, val_data, train_config)
        train_time = time.time() - start_time

        test_preds = trainer.generate_predictions(
            train_result["model"], test_ds,
            batch_size=8,
            max_length=cfg["generation"].get("max_length", cfg["model"].get("max_length", 128)),
            num_beams=cfg["generation"]["num_beams"],
        )
        test_metrics = evaluator.evaluate_all(
            [s.source for s in test_ds.samples],
            test_preds,
            [s.target  for s in test_ds.samples],
        )

        entry = {
            "method": method,
            "data_size": size,
            "seed": seed,
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "test_bleu": test_metrics["bleu"],
            "test_chrf":  test_metrics["chrf"],
            "training_time_seconds": train_time,
            "learning_rate": train_config["learning_rate"],
            "fp16": train_config["fp16"],
        }

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f" val BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")
        logger.info(f" test BLEU={test_metrics['bleu']:.4f}  chrF={test_metrics['chrf']:.2f}")
        logger.info(f" time {train_time/60:.1f}m")

        return entry

    except Exception as e:
        logger.error(f" FAILED: {e}\n{traceback.format_exc()}")
        return {
            "method": method,
            "data_size": size,
            "seed": seed,
            "val_bleu": 0.0,
            "val_chrf": 0.0,
            "val_loss": 999.0,
            "test_bleu": 0.0,
            "test_chrf": 0.0,
            "training_time_seconds": 0,
            "failed": True,
        }

    finally:
        del trainer
        torch.cuda.empty_cache()
        gc.collect()


def run_all(methods, data_sizes, seeds, cfg, train_ds, val_ds, test_ds, evaluator, output_dir, logger):
    results = []
    total   = len(methods) * len(data_sizes) * len(seeds)
    run_idx = 0

    for method in methods:
        logger.info(f"\n{'='*60}\nMETHOD: {method.upper()}\n{'='*60}")
        for size in data_sizes:
            for seed in seeds:
                run_idx   += 1
                size_label = "full" if size == len(train_ds) else str(size)

                # resume: skip if already done
                run_dir   = output_dir / method / f"size{size_label}_seed{seed}"
                done_file = run_dir / "metrics.json"
                if done_file.exists():
                    logger.info(f"[{run_idx}/{total}] skipping {method} size={size_label} seed={seed}")
                    with open(done_file) as f:
                        results.append(json.load(f))
                    continue

                logger.info(f"\n[{run_idx}/{total}] {method} size={size_label} seed={seed}")
                entry = run_one(method, size, seed, cfg, train_ds, val_ds, test_ds,
                                evaluator, output_dir, logger)
                results.append(entry)

    return results


def load_existing_results(output_dir: Path) -> list:
    # collect results already saved from previous runs (other method's job)
    results = []
    for metrics_file in output_dir.glob("*/size*/metrics.json"):
        with open(metrics_file) as f:
            results.append(json.load(f))
    return results


def save_report(output_dir, logger):
    all_results = load_existing_results(output_dir)
    if not all_results:
        logger.warning("No results found.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    valid_df = df[~df["failed"].fillna(False)] if "failed" in df.columns else df
    if not valid_df.empty:
        summary = valid_df.groupby(["method", "data_size"]).agg(
            test_bleu_mean=("test_bleu", "mean"),
            test_bleu_std=("test_bleu", "std"),
            test_chrf_mean=("test_chrf", "mean"),
            time_mean=("training_time_seconds", "mean"),
        ).round(4)
        logger.info(f"\n{summary}")
        summary.to_csv(output_dir / "summary.csv")

    logger.info(f"Results saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "ft", "all"],
        default="all",
        help="Which method to run. Use 'all' to run both sequentially.",
    )
    return parser.parse_args()


def main():
    # 1. load config
    args = parse_args()
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp6_method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir)
    logger.info("LoRA vs FT")

    # 2. load data
    data_manager = DataManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_splits()
    evaluator  = BaseEvaluator(use_comet=False)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 3. build experiment grid
    methods    = ["lora", "ft"] if args.method == "all" else [args.method]
    data_sizes = build_data_sizes(train_ds)
    seeds      = cfg.get("experiment", {}).get("seeds", [42, 123, 456])
    total      = len(methods) * len(data_sizes) * len(seeds)
    logger.info(f"Method: {methods} | Sizes: {data_sizes} | Seeds: {seeds} | Total: {total}")

    # 4. run experiments
    run_all(methods, data_sizes, seeds, cfg,
            train_ds, val_ds, test_ds, evaluator, output_dir, logger)
    
    # 5. save results
    save_report(output_dir, logger)


if __name__ == "__main__":
    main()
