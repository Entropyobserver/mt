import gc
import json
import logging
import sys
import traceback
from pathlib import Path

import optuna
import torch
import yaml
from optuna.pruners import SuccessiveHalvingPruner

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data.data_loader import DataManager
from scripts.model.lora_trainer import LoRATrainer


def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp3_stage1")
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


def load_data(cfg, logger):
    data_manager = DataManager(cfg)
    train_ds, val_ds, _ = data_manager.load_splits()

    stage1_cfg = cfg.get("experiment", {}).get("stage1", {})
    train_size = stage1_cfg.get("train_size", 2000)
    train_ds = train_ds.subset(train_size, seed=cfg["project"]["seed"])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds


def build_train_config(cfg, r, alpha, dropout, trial_dir) -> dict:
    stage1_cfg = cfg.get("experiment", {}).get("stage1", {})

    return {
        "output_dir": str(trial_dir / "training"),
        "seed": cfg["project"]["seed"],
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": cfg["lora"]["target_modules"],
        "epochs": stage1_cfg.get("epochs", cfg["training"]["epochs"]),
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": stage1_cfg.get("eval_steps", cfg["training"]["eval_steps"]),
        "early_stopping_patience": stage1_cfg.get("early_stopping_patience", 1),
        "fp16": cfg["training"]["fp16"],
        "max_length": cfg["model"].get("max_length", 128),
        "generation_max_length": cfg["generation"].get("max_length", cfg["model"].get("max_length", 128)),
        "generation_num_beams": cfg["generation"]["num_beams"],
        "save_total_limit": 1,
        "save_final_model": False,
    }


def sample_params(trial, cfg):
    stage1_cfg = cfg.get("experiment", {}).get("stage1", {})
    r_values = stage1_cfg.get("r", [8, 16, 32])
    alpha_values = stage1_cfg.get("alpha", [16, 32, 64])
    dropout_min = stage1_cfg.get("dropout_min", 0.0)
    dropout_max = stage1_cfg.get("dropout_max", 0.2)
    dropout_step = stage1_cfg.get("dropout_step", 0.05)

    return {
        "r": trial.suggest_categorical("r", r_values),
        "alpha": trial.suggest_categorical("alpha", alpha_values),
        "dropout": trial.suggest_float(
            "dropout",
            dropout_min,
            dropout_max,
            step=dropout_step,
        ),
    }


def make_objective(cfg, train_data, val_data, output_dir, logger):
    def objective(trial):
        params = sample_params(trial, cfg)
        logger.info(
            f"Trial {trial.number}: r={params['r']}, "
            f"alpha={params['alpha']}, dropout={params['dropout']}"
        )

        trial_dir = output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trainer = LoRATrainer(
            model_name=cfg["model"]["pretrained"],
            src_lang=cfg["model"]["src_lang"],
            tgt_lang=cfg["model"]["tgt_lang"],
        )
        train_config = build_train_config(cfg, **params, trial_dir=trial_dir)

        try:
            result = trainer.train(train_data, val_data, train_config)
            logger.info(f"BLEU={result['bleu']:.4f}  chrF={result['chrf']:.2f}")

            with open(trial_dir / "result.json", "w") as f:
                json.dump(
                    {
                        "trial": trial.number,
                        "params": params,
                        "bleu": result["bleu"],
                        "chrf": result["chrf"],
                        "loss": result["loss"],
                    },
                    f,
                    indent=2,
                )

            return result["bleu"]

        except Exception as e:
            logger.error(f"Trial failed: {e}\n{traceback.format_exc()}")
            return 0.0

        finally:
            if "result" in locals():
                del result
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return objective


def run_study(cfg, train_data, val_data, output_dir, logger):
    num_trials = cfg.get("experiment", {}).get("num_trials", 50)
    logger.info(f"Trials={num_trials}, Pruner=SuccessiveHalving")

    study = optuna.create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
        study_name="nllb_lora_stage1",
    )
    study.optimize(
        make_objective(cfg, train_data, val_data, output_dir, logger),
        n_trials=num_trials,
        show_progress_bar=True,
    )
    return study


def save_report(study, output_dir, logger):
    study.trials_dataframe().to_csv(output_dir / "results.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(f"Completed trials: {len(completed)}")

    best = study.best_trial
    logger.info(
        f"BEST: r={best.params['r']}, alpha={best.params['alpha']}, "
        f"dropout={best.params['dropout']}, BLEU={best.value:.4f}"
    )

    top_configs = [
        {
            "trial": trial.number,
            "r": trial.params["r"],
            "alpha": trial.params["alpha"],
            "dropout": trial.params["dropout"],
            "bleu": trial.value,
        }
        for trial in sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    ]

    with open(output_dir / "top_configs.json", "w") as f:
        json.dump(top_configs, f, indent=2)


def main():
    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp3_optuna_stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 3 - OPTUNA STAGE 1")

    train_ds, val_ds = load_data(cfg, logger)
    train_data = [sample.to_dict() for sample in train_ds.samples]
    val_data = [sample.to_dict() for sample in val_ds.samples]

    study = run_study(cfg, train_data, val_data, output_dir, logger)
    save_report(study, output_dir, logger)


if __name__ == "__main__":
    main()
