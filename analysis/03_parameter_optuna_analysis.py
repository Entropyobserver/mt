import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "outputs" / "exp3_optuna_stage1" / "results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp3_optuna_stage1" / "figures"


def is_pareto_efficient(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, score in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] > score, axis=1)
            is_efficient[i] = True
    return is_efficient


def normalize_optuna_csv(input_file):
    df = pd.read_csv(input_file)
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"].copy()

    if "bleu" not in df.columns:
        if "value" in df.columns:
            df["bleu"] = df["value"]
        elif "values_0" in df.columns:
            df["bleu"] = df["values_0"]
        else:
            raise KeyError("Could not find BLEU column in Optuna CSV")

    if "chrf" not in df.columns:
        if "values_1" in df.columns:
            df["chrf"] = -df["values_1"]
        else:
            df["chrf"] = np.nan

    if df["chrf"].isna().all():
        df = merge_trial_json_metrics(df, Path(input_file).parent)

    required = ["params_alpha", "params_r", "params_dropout", "bleu"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df


def merge_trial_json_metrics(df, trial_root):
    rows = []
    for result_file in sorted(Path(trial_root).glob("trial_*/result.json")):
        with result_file.open(encoding="utf-8") as f:
            result = json.load(f)
        rows.append(
            {
                "number": result.get("trial"),
                "chrf_from_json": result.get("chrf"),
                "loss_from_json": result.get("loss"),
            }
        )

    if not rows or "number" not in df.columns:
        return df

    metric_df = pd.DataFrame(rows)
    merged = df.merge(metric_df, on="number", how="left")
    merged["chrf"] = merged["chrf"].fillna(merged["chrf_from_json"])
    if "loss" not in merged.columns:
        merged["loss"] = merged["loss_from_json"]
    return merged.drop(columns=[col for col in ["chrf_from_json", "loss_from_json"] if col in merged])


def plot_parameter_importance(df, output_dir):
    x = df[["params_alpha", "params_r", "params_dropout"]]
    y = df["bleu"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x, y)
    importances = rf.feature_importances_

    plt.figure(figsize=(8, 5))
    params = ["LoRA Alpha", "LoRA Rank", "Dropout"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    bars = plt.bar(params, importances, color=colors, edgecolor="black", linewidth=1.2)
    plt.ylabel("Importance Score", fontsize=12, fontweight="bold")
    plt.title("Hyperparameter Importance from Optuna Study", fontsize=14, fontweight="bold")
    plt.ylim(0, max(importances) * 1.15 if max(importances) > 0 else 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    output_path = output_dir / "optuna_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Importance plot saved to {output_path}")

    return dict(zip(params, importances))


def plot_pareto(df, output_dir):
    if df["chrf"].isna().all():
        print("Skipping Pareto plot because chrF is not available.")
        return None

    scores = np.column_stack([df["bleu"], df["chrf"]])
    pareto_mask = is_pareto_efficient(scores)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df.loc[~pareto_mask, "bleu"],
        df.loc[~pareto_mask, "chrf"],
        c="lightgray",
        s=50,
        alpha=0.6,
        label="Other Trials",
        edgecolors="gray",
    )
    plt.scatter(
        df.loc[pareto_mask, "bleu"],
        df.loc[pareto_mask, "chrf"],
        c="red",
        s=100,
        alpha=0.8,
        label="Pareto Front",
        edgecolors="darkred",
        linewidth=1.5,
    )

    best_idx = df["bleu"].idxmax()
    plt.scatter(
        df.loc[best_idx, "bleu"],
        df.loc[best_idx, "chrf"],
        c="gold",
        s=300,
        marker="*",
        label="Best BLEU",
        edgecolors="black",
        linewidth=2,
        zorder=5,
    )

    plt.xlabel("BLEU Score", fontsize=12, fontweight="bold")
    plt.ylabel("chrF Score", fontsize=12, fontweight="bold")
    plt.title("Pareto Front for BLEU and chrF Scores", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    output_path = output_dir / "pareto_front.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Pareto plot saved to {output_path}")
    return pareto_mask


def plot_top_configs(input_file, output_dir):
    with Path(input_file).open(encoding="utf-8") as f:
        configs = json.load(f)

    if isinstance(configs, dict):
        configs = [configs]
    if not configs:
        raise ValueError(f"No configurations found in {input_file}")

    labels = [
        f"r={c.get('r')}, alpha={c.get('alpha')}, d={c.get('dropout')}"
        for c in configs
    ]
    bleus = [c.get("bleu", c.get("val_bleu")) for c in configs]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(len(configs)), bleus, color="#2E86AB", edgecolor="black", linewidth=1.2)
    plt.xticks(range(len(configs)), labels, rotation=20, ha="right")
    plt.ylabel("BLEU Score", fontsize=12, fontweight="bold")
    plt.title("Top Optuna Stage 1 Configurations", fontsize=14, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, bleu in zip(bars, bleus):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bleu:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    output_path = output_dir / "top_configs.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Top-config plot saved to {output_path}")
    return configs


def analyze_csv(input_file, output_dir):
    df = normalize_optuna_csv(input_file)
    importances = plot_parameter_importance(df, output_dir)
    pareto_mask = plot_pareto(df, output_dir)
    best_idx = df["bleu"].idxmax()
    best = df.loc[best_idx]

    print("Data check:")
    print(f"Trials: {len(df)}")
    print(f"BLEU range: [{df['bleu'].min():.4f}, {df['bleu'].max():.4f}]")
    if not df["chrf"].isna().all():
        print(f"chrF range: [{df['chrf'].min():.4f}, {df['chrf'].max():.4f}]")
    print("Parameter importances:")
    for param, importance in importances.items():
        print(f"  {param}: {importance:.4f}")
    if pareto_mask is not None:
        print(f"Pareto front: {pareto_mask.sum()} configurations")
    print("Best BLEU config:")
    print(f"  r={best['params_r']}, alpha={best['params_alpha']}, dropout={best['params_dropout']}")
    print(f"  BLEU={best['bleu']:.4f}")
    if not pd.isna(best["chrf"]):
        print(f"  chrF={best['chrf']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file.suffix.lower() == ".csv":
        analyze_csv(args.input_file, args.output_dir)
    else:
        configs = plot_top_configs(args.input_file, args.output_dir)
        best = max(configs, key=lambda item: item.get("bleu", item.get("val_bleu", float("-inf"))))
        print("Top-config summary:")
        print(f"Configs: {len(configs)}")
        print(f"Best: r={best.get('r')}, alpha={best.get('alpha')}, dropout={best.get('dropout')}")
        print(f"BLEU={best.get('bleu', best.get('val_bleu')):.4f}")


if __name__ == "__main__":
    main()
