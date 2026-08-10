import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "outputs" / "exp5_final_eval" / "all_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp5_final_eval" / "figures"


def load_results(input_file):
    with Path(input_file).open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "metrics" in data:
        config = data.get("configuration", {})
        metrics = data["metrics"]
        return [
            {
                "seed": data.get("seed", "single"),
                "configuration": config,
                "test_bleu": metrics.get("bleu"),
                "test_chrf": metrics.get("chrf"),
                "test_comet": metrics.get("comet"),
            }
        ]

    if isinstance(data, list):
        return [row for row in data if not row.get("failed")]

    raise ValueError(f"Unsupported final-eval result format: {input_file}")


def summarize(rows):
    metrics = ["test_bleu", "test_chrf", "test_comet"]
    summary = {}
    for metric in metrics:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    config = rows[0].get("configuration", {}) if rows else {}
    return config, summary


def plot_metrics(summary, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [("test_bleu", "BLEU"), ("test_chrf", "chrF"), ("test_comet", "COMET")]
    available = [(key, name) for key, name in order if key in summary]

    names = [name for _, name in available]
    means = [summary[key]["mean"] for key, _ in available]
    stds = [summary[key]["std"] for key, _ in available]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, color=["#2E86AB", "#A23B72", "#F18F01"], capsize=5)
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title(f"Final Evaluation Metrics ({label})", fontsize=14, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.4f}" if mean < 10 else f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / "final_eval_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Final-eval plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="new")
    args = parser.parse_args()

    rows = load_results(args.input_file)
    config, summary = summarize(rows)
    plot_metrics(summary, args.output_dir, args.label)

    print("Final evaluation summary:")
    print(f"Configuration: {config}")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f} (n={values['n']})")


if __name__ == "__main__":
    main()
