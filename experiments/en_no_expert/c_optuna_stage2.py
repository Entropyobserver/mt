import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data.data_loader import DataManager
from scripts.model.lora_trainer import LoRATrainer


def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp3_stage2")
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


def load_top_configs(cfg, logger) -> list:
    stage1_dir = project_root / cfg["paths"]["output_dir"] / "exp3_optuna_stage1"
    top_configs_file = stage1_dir / "top_configs.json"

    if not top_configs_file.exists():
        logger.error(f"top_configs.json not found at {top_configs_file}")
        return []

    with open(top_configs_file, encoding="utf-8") as f:
        top_configs = json.load(f)

    logger.info(f"Loaded {len(top_configs)} configurations from Stage 1")
    return top_configs


def load_data(cfg, logger):
    data_manager = DataManager(cfg)
    train_ds, val_ds, _ = data_manager.load_splits()

    stage2_cfg = cfg.get("experiment", {}).get("stage2", {})
    train_size = stage2_cfg.get("train_size", 8000)
    train_ds = train_ds.subset(train_size, seed=cfg["project"]["seed"])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


def build_train_config(cfg, config, seed, run_dir) -> dict:
    stage2_cfg = cfg.get("experiment", {}).get("stage2", {})

    return {
        "output_dir": str(run_dir / "training"),
        "seed": seed,
        "r": config["r"],
        "alpha": config["alpha"],
        "dropout": config["dropout"],
        "target_modules": cfg["lora"]["target_modules"],
        "epochs": stage2_cfg.get("epochs", cfg["training"]["epochs"]),
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": stage2_cfg.get("eval_steps", cfg["training"]["eval_steps"]),
        "early_stopping_patience": stage2_cfg.get(
            "early_stopping_patience",
            cfg["training"]["early_stopping_patience"],
        ),
        "fp16": cfg["training"]["fp16"],
        "max_length": cfg["model"].get("max_length", 128),
        "generation_max_length": cfg["generation"].get("max_length", cfg["model"].get("max_length", 128)),
        "generation_num_beams": cfg["generation"]["num_beams"],
        "save_total_limit": 1,
        "save_final_model": True,
    }


def run_one(config, config_idx, run, cfg, train_data, val_data, output_dir, logger):
    seed = cfg["project"]["seed"] + run
    run_dir = output_dir / f"config_{config_idx}_run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )
    train_config = build_train_config(cfg, config, seed, run_dir)

    try:
        config_start = time.time()
        train_result = trainer.train(train_data, val_data, train_config)
        config_time = time.time() - config_start

        entry = {
            "config_id": config_idx,
            "run": run,
            "seed": seed,
            "r": config["r"],
            "alpha": config["alpha"],
            "dropout": config["dropout"],
            "stage1_bleu": config["bleu"],
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "model_path": train_result.get("final_model_path", ""),
            "training_time": config_time,
        }

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "config": config,
                    "run": run,
                    "seed": seed,
                    "val_metrics": {
                        "bleu": train_result["bleu"],
                        "chrf": train_result["chrf"],
                        "loss": train_result["loss"],
                    },
                    "training_time": config_time,
                },
                f,
                indent=2,
            )

        logger.info(f" val BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")
        logger.info(f" time {config_time/60:.1f}m")

        return entry

    except Exception as e:
        logger.error(f"Run failed: {e}\n{traceback.format_exc()}")
        return {
            "config_id": config_idx,
            "run": run,
            "seed": seed,
            "failed": True,
            "val_bleu": 0.0,
            "training_time": 0,
        }

    finally:
        if "train_result" in locals():
            del train_result
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_all(top_configs, cfg, train_data, val_data, output_dir, logger):
    stage2_cfg = cfg.get("experiment", {}).get("stage2", {})
    num_runs = stage2_cfg.get("runs", 3)
    results = []

    logger.info(f"Configs={len(top_configs)} | Runs per config={num_runs}")

    for config_idx, config in enumerate(top_configs):
        logger.info(
            f"\n[Config {config_idx + 1}] r={config['r']} "
            f"alpha={config['alpha']} dropout={config['dropout']}"
        )

        for run in range(num_runs):
            logger.info(f"  Run {run + 1}/{num_runs}")
            entry = run_one(
                config,
                config_idx,
                run,
                cfg,
                train_data,
                val_data,
                output_dir,
                logger,
            )
            results.append(entry)

    return results


def save_report(results, output_dir, logger):
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_df = df[~df["failed"].fillna(False)] if "failed" in df.columns else df

    if valid_df.empty:
        logger.info("No successful runs to summarize.")
        return

    summary = valid_df.groupby("config_id").agg(
        val_bleu_mean=("val_bleu", "mean"),
        val_bleu_std=("val_bleu", "std"),
        val_chrf_mean=("val_chrf", "mean"),
        val_loss_mean=("val_loss", "mean"),
    ).round(4)
    logger.info(f"\n{summary}")
    summary.to_csv(output_dir / "summary.csv")

    best_config_id = summary["val_bleu_mean"].idxmax()
    best_rows = valid_df[valid_df["config_id"] == best_config_id]
    best_row = best_rows.sort_values("val_bleu", ascending=False).iloc[0]

    total_time = valid_df["training_time"].sum()
    logger.info(f"Total time: {total_time/3600:.2f}h")
    logger.info(
        f"\nBEST CONFIG BY VALIDATION BLEU: r={best_row['r']} "
        f"alpha={best_row['alpha']} dropout={best_row['dropout']} "
        f"mean BLEU={summary.loc[best_config_id, 'val_bleu_mean']:.4f}"
    )

    with open(output_dir / "best_config.json", "w") as f:
        json.dump(
            {
                "config_id": int(best_config_id),
                "r": int(best_row["r"]),
                "alpha": int(best_row["alpha"]),
                "dropout": float(best_row["dropout"]),
                "val_bleu_mean": float(summary.loc[best_config_id, "val_bleu_mean"]),
                "val_bleu_std": float(summary.loc[best_config_id, "val_bleu_std"]),
                "model_path": str(best_row["model_path"]),
            },
            f,
            indent=2,
        )


def main():
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp3_optuna_stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 3: STAGE 2 VALIDATION")

    top_configs = load_top_configs(cfg, logger)
    if not top_configs:
        return

    train_ds, val_ds = load_data(cfg, logger)
    train_data = [sample.to_dict() for sample in train_ds.samples]
    val_data = [sample.to_dict() for sample in val_ds.samples]

    results = run_all(top_configs, cfg, train_data, val_data, output_dir, logger)
    save_report(results, output_dir, logger)


if __name__ == "__main__":
    main()
