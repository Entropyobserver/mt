import gc
import json
import logging
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import yaml

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data.data_loader import DataManager
from scripts.model.lora_trainer import LoRATrainer


def get_logger(output_dir: Path, log_file: str = "experiment.log") -> logging.Logger:
    logger = logging.getLogger("exp2")
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


def get_recommended_size(cfg, logger) -> int:
    size = cfg.get("experiment", {}).get("gridsearch", {}).get("train_size", 8000)
    logger.info(f"Using exp1-determined training size: {size}")
    return size


def build_param_grid(cfg) -> list:
    gs = cfg.get("experiment", {}).get("gridsearch", {})
    r_values = gs.get("r", [8, 16, 32])
    alpha_values = gs.get("alpha", [16, 32, 64])
    dropout_values = gs.get("dropout", [0.0, 0.1, 0.2])
    return list(product(r_values, alpha_values, dropout_values))


def build_train_config(cfg, r, alpha, dropout, config_output_dir) -> dict:
    return {
        "output_dir": str(config_output_dir / "training"),
        "seed": cfg["project"]["seed"],
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": cfg["lora"]["target_modules"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16": cfg["training"]["fp16"],
        "max_length": cfg["model"].get("max_length", 128),
        "generation_max_length": cfg["generation"].get("max_length", cfg["model"].get("max_length", 128)),
        "generation_num_beams": cfg["generation"]["num_beams"],
        "save_total_limit": 1,
        "save_final_model": True,
    }


def run_one_config(
    r,
    alpha,
    dropout,
    cfg,
    train_data,
    val_data,
    output_dir,
    logger,
):
    config_name = f"r{r}_a{alpha}_d{dropout}"
    config_output_dir = output_dir / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    trainer = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )

    train_config = build_train_config(cfg, r, alpha, dropout, config_output_dir)

    try:
        config_start = time.time()
        train_result = trainer.train(train_data, val_data, train_config)
        config_time = time.time() - config_start

        entry = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "model_path": train_result.get("final_model_path", ""),
            "training_time_seconds": config_time,
        }

        with open(config_output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "config": {"r": r, "alpha": alpha, "dropout": dropout},
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
        logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")
        with open(config_output_dir / "error.log", "w") as f:
            f.write(f"{e}\n{traceback.format_exc()}")

        return {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "val_bleu": 0.0,
            "val_chrf": 0.0,
            "val_loss": 999.0,
            "model_path": "",
            "training_time_seconds": 0,
            "failed": True,
        }

    finally:
        if "train_result" in locals():
            del train_result
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_grid(all_configs, cfg, train_data, val_data, output_dir, logger):
    results = []
    best_val_bleu = 0.0
    best_config = None
    best_model_path = None

    total = len(all_configs)

    for idx, (r, alpha, dropout) in enumerate(all_configs, 1):
        logger.info(f"\n[{idx}/{total}] r={r}, alpha={alpha}, dropout={dropout}")

        entry = run_one_config(
            r,
            alpha,
            dropout,
            cfg,
            train_data,
            val_data,
            output_dir,
            logger,
        )
        results.append(entry)

        if entry["val_bleu"] > best_val_bleu:
            best_val_bleu = entry["val_bleu"]
            best_config = {"r": r, "alpha": alpha, "dropout": dropout}
            best_model_path = entry["model_path"]

    return results, best_config, best_val_bleu, best_model_path


def generate_final_report(
    output_dir,
    logger,
    results,
    best_config=None,
    best_val_bleu=None,
    best_model_path=None,
):
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_df = df[~df["failed"].fillna(False)] if "failed" in df.columns else df

    if not valid_df.empty:
        top5 = valid_df.nlargest(5, "val_bleu")
        logger.info("\nTop 5 configurations by validation BLEU:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            logger.info(
                f"  {i}. r={row['r']}, alpha={row['alpha']}, dropout={row['dropout']}  "
                f"val BLEU={row['val_bleu']:.4f}"
            )

        if not best_config:
            best_row = valid_df.loc[valid_df["val_bleu"].idxmax()]
            best_config = {
                "r": int(best_row["r"]),
                "alpha": int(best_row["alpha"]),
                "dropout": float(best_row["dropout"]),
            }
            best_val_bleu = float(best_row["val_bleu"])
            best_model_path = best_row["model_path"]

    if best_config:
        logger.info(
            f"\nBest by validation BLEU: r={best_config['r']}, "
            f"alpha={best_config['alpha']}, dropout={best_config['dropout']}, "
            f"val BLEU={best_val_bleu:.4f}"
        )

        with open(output_dir / "best_config.json", "w") as f:
            json.dump(
                {
                    **best_config,
                    "val_bleu": best_val_bleu,
                    "model_path": str(best_model_path),
                },
                f,
                indent=2,
            )

    total_time = sum(r.get("training_time_seconds", 0) for r in results)
    logger.info(f"Total time: {total_time/3600:.2f}h")
    logger.info(f"Results saved to {output_dir}")


def main():
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp2_gridsearch"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 2: PARAMETER SENSITIVITY ANALYSIS")

    data_manager = DataManager(cfg)
    train_ds, val_ds, _ = data_manager.load_splits()

    size = get_recommended_size(cfg, logger)
    train_ds = train_ds.subset(size, seed=cfg["project"]["seed"])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    all_configs = build_param_grid(cfg)
    logger.info(f"Total configs: {len(all_configs)}")

    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]

    results, best_config, best_val_bleu, best_model_path = run_grid(
        all_configs,
        cfg,
        train_data,
        val_data,
        output_dir,
        logger,
    )

    generate_final_report(
        output_dir,
        logger,
        results,
        best_config,
        best_val_bleu,
        best_model_path,
    )


if __name__ == "__main__":
    main()
