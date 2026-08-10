import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "outputs" / "exp4_optuna_stage2" / "results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp4_optuna_stage2" / "figures"


def summarize(results_file):
    df = pd.read_csv(results_file)
    run_col = "run_id" if "run_id" in df.columns else "run"
    summary = (
        df.groupby(["config_id", "r", "alpha", "dropout"])
        .agg(
            val_bleu_mean=("val_bleu", "mean"),
            val_bleu_std=("val_bleu", "std"),
            val_chrf_mean=("val_chrf", "mean"),
            val_loss_mean=("val_loss", "mean"),
            runs=(run_col, "count"),
        )
        .reset_index()
        .sort_values("val_bleu_mean", ascending=False)
    )
    return df, summary


def plot_stage2(summary, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"C{int(row.config_id)}\nr={int(row.r)}, a={int(row.alpha)}, d={row.dropout:g}"
        for row in summary.itertuples()
    ]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(
        range(len(summary)),
        summary["val_bleu_mean"],
        yerr=summary["val_bleu_std"].fillna(0),
        color="#2E86AB",
        edgecolor="black",
        linewidth=1.2,
        capsize=5,
    )
    plt.xticks(range(len(summary)), labels)
    plt.ylabel("Validation BLEU", fontsize=12, fontweight="bold")
    plt.title(f"Optuna Stage 2 Validation ({label})", fontsize=14, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    lower = max(0, summary["val_bleu_mean"].min() - 0.01)
    upper = summary["val_bleu_mean"].max() + 0.01
    plt.ylim(lower, upper)

    for bar, row in zip(bars, summary.itertuples()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            row.val_bleu_mean,
            f"{row.val_bleu_mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / "stage2_validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Stage 2 validation plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="new")
    args = parser.parse_args()

    _, summary = summarize(args.results_file)
    plot_stage2(summary, args.output_dir, args.label)

    print("Stage 2 summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
