import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "outputs" / "exp2_gridsearch" / "new_results" / "results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp2_gridsearch" / "new_results" / "figures"


def create_parameter_sensitivity_plots(results_file=DEFAULT_RESULTS_FILE, output_dir=DEFAULT_OUTPUT_DIR):
    results_file = Path(results_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_file.open(encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    dropout_values = sorted(df["dropout"].unique())

    fig, axes = plt.subplots(1, len(dropout_values), figsize=(18, 5))
    if len(dropout_values) == 1:
        axes = [axes]

    for idx, dropout in enumerate(dropout_values):
        subset = df[df["dropout"] == dropout]
        pivot = subset.pivot_table(values="val_bleu", index="alpha", columns="r", aggfunc="mean")

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            cbar_kws={"label": "BLEU Score"},
            ax=axes[idx],
            linewidths=0.5,
            linecolor="gray",
            vmin=0.545,
            vmax=0.625,
        )
        axes[idx].set_title(f"Dropout = {dropout}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("LoRA Rank (r)", fontsize=12)
        axes[idx].set_ylabel("LoRA Alpha (alpha)", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "hyperparameter_heatmaps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmaps saved to {output_path}")

    create_parallel_coordinates(df, output_dir)
    create_parameter_importance_plot(df, output_dir)


def create_parallel_coordinates(df, output_dir):
    from pandas.plotting import parallel_coordinates

    df_plot = df[["r", "alpha", "dropout", "val_bleu"]].copy()
    df_plot["performance"] = pd.cut(df_plot["val_bleu"], bins=3, labels=["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(
        df_plot,
        "performance",
        color=["#C73E1D", "#F18F01", "#2E86AB"],
        linewidth=2,
        alpha=0.7,
        ax=ax,
    )

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Parallel Coordinates: Hyperparameter Space Exploration", fontsize=14, fontweight="bold")
    ax.legend(title="BLEU Performance", loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "parallel_coordinates.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Parallel coordinates plot saved to {output_path}")


def create_parameter_importance_plot(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    params = ["r", "alpha", "dropout"]
    param_labels = ["LoRA Rank (r)", "LoRA Alpha (alpha)", "Dropout Rate"]

    for param, label, ax in zip(params, param_labels, axes):
        grouped = df.groupby(param)["val_bleu"].agg(["mean", "std"])
        x = grouped.index
        y = grouped["mean"]
        yerr = grouped["std"]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            linewidth=2.5,
            markersize=10,
            capsize=5,
            capthick=2,
            color="#2E86AB",
        )
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("BLEU Score", fontsize=12)
        ax.set_title(f"Effect of {label}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        best_val = y.max()
        best_param = x[y.argmax()]
        ax.axhline(y=best_val, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.plot(best_param, best_val, "r*", markersize=20, zorder=5)
        ax.annotate(
            f"Best: {best_param}\n{best_val:.4f}",
            xy=(best_param, best_val),
            xytext=(best_param, best_val * 0.998),
            fontsize=10,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()
    output_path = Path(output_dir) / "parameter_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Parameter importance plot saved to {output_path}")

    print_best_config(df)


def print_best_config(df):
    best_idx = df["val_bleu"].idxmax()
    best_config = df.loc[best_idx]
    print("\n" + "=" * 50)
    print("BEST CONFIGURATION:")
    print("=" * 50)
    print(f"LoRA Rank (r): {best_config['r']}")
    print(f"LoRA Alpha (alpha): {best_config['alpha']}")
    print(f"Dropout: {best_config['dropout']}")
    print(f"BLEU: {best_config['val_bleu']:.4f}")
    print(f"chrF: {best_config['val_chrf']:.2f}")
    print(f"Loss: {best_config['val_loss']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    create_parameter_sensitivity_plots(args.results_file, args.output_dir)
