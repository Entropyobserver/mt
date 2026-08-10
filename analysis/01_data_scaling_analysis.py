import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "outputs" / "exp1_data_scaling" / "new_results" / "results.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "exp1_data_scaling" / "new_results" / "figures" / "data_scaling_analysis.png"


def pick_field(row, candidates):
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(f"None of these fields found: {candidates}")


def load_and_group(results_file):
    with open(results_file) as f:
        results = json.load(f)

    grouped = defaultdict(lambda: {"bleu": [], "chrf": []})
    valid_results = [r for r in results if not r.get("failed")]
    if not valid_results:
        raise ValueError(f"No valid results found in {results_file}")

    first = valid_results[0]
    size_field = pick_field(first, ["data_size", "actual_size", "sample_size"])
    bleu_field = pick_field(first, ["test_bleu", "val_bleu"])
    chrf_field = pick_field(first, ["test_chrf", "val_chrf"])

    for r in results:
        if r.get("failed"):
            continue
        size = r[size_field]
        grouped[size]["bleu"].append(r[bleu_field])
        grouped[size]["chrf"].append(r[chrf_field])

    sizes = sorted(grouped.keys())
    bleus = [np.mean(grouped[s]["bleu"]) for s in sizes]
    chrfs = [np.mean(grouped[s]["chrf"]) for s in sizes]
    bleu_stds = [np.std(grouped[s]["bleu"]) for s in sizes]
    chrf_stds = [np.std(grouped[s]["chrf"]) for s in sizes]
    metric_label = "test" if bleu_field.startswith("test_") else "validation"
    return sizes, bleus, chrfs, bleu_stds, chrf_stds, metric_label


def compute_marginal_gains(sizes, bleus):
    gains = [0]
    for i in range(1, len(sizes)):
        gain = bleus[i] - bleus[i - 1]
        size_diff = sizes[i] - sizes[i - 1]
        gains.append(gain / (size_diff / 1000))
    return gains


def find_optimal(sizes, bleus, threshold=0.95):
    best = max(bleus)
    for i, bleu in enumerate(bleus):
        if bleu >= best * threshold:
            return i, best
    return None, best


def plot_quality(ax, sizes, bleus, chrfs, bleu_stds, chrf_stds, optimal_idx, best_bleu, metric_label):
    color1 = "#2E86AB"
    color2 = "#A23B72"

    ax.errorbar(sizes, bleus, yerr=bleu_stds, fmt="o-", linewidth=2.5, markersize=8,
                color=color1, capsize=5, capthick=2, label="BLEU")
    ax.set_xlabel("Training Data Size", fontsize=12)
    ax.set_ylabel("BLEU Score", color=color1, fontsize=12)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=best_bleu * 0.95, color="red", linestyle="--", alpha=0.6,
               linewidth=1.5, label="95% BLEU threshold")

    if optimal_idx is not None:
        opt_size = sizes[optimal_idx]
        opt_bleu = bleus[optimal_idx]
        ax.plot(opt_size, opt_bleu, "r*", markersize=20, zorder=5)
        ax.annotate(
            f"Optimal Point\n{opt_size} samples\n{opt_bleu/best_bleu*100:.1f}% of max",
            xy=(opt_size, opt_bleu),
            xytext=(opt_size * 0.6, opt_bleu * 0.92),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="red", lw=2),
        )

    ax2 = ax.twinx()
    ax2.errorbar(sizes, chrfs, yerr=chrf_stds, fmt="s--", linewidth=2.5, markersize=8,
                 color=color2, capsize=5, capthick=2, label="chrF")
    ax2.set_ylabel("chrF Score", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)
    ax.set_title(
        f"Translation Quality vs Training Data Size ({metric_label})",
        fontsize=14,
        fontweight="bold",
    )


def plot_marginal(ax, sizes, gains):
    from matplotlib.patches import Patch

    colors = []
    for g in gains:
        if g > 0.05: colors.append("#2E86AB")
        elif g > 0.01: colors.append("#4A9EBD")
        elif g > 0.005: colors.append("#F18F01")
        else: colors.append("#C73E1D")

    ax.bar(range(len(sizes)), gains, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.set_xlabel("Training Data Size", fontsize=12)
    ax.set_ylabel("BLEU Gain per 1,000 Samples", fontsize=12)
    ax.set_title("Marginal Efficiency Analysis", fontsize=14, fontweight="bold")
    ax.axhline(y=0.005, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor="#2E86AB", alpha=0.7, edgecolor="black", label="High efficiency (>0.05)"),
        Patch(facecolor="#4A9EBD", alpha=0.7, edgecolor="black", label="Medium efficiency (0.01-0.05)"),
        Patch(facecolor="#F18F01", alpha=0.7, edgecolor="black", label="Low efficiency (0.005-0.01)"),
        Patch(facecolor="#C73E1D", alpha=0.7, edgecolor="black", label="Diminishing returns (<0.005)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    sizes, bleus, chrfs, bleu_stds, chrf_stds, metric_label = load_and_group(args.results_file)
    gains = compute_marginal_gains(sizes, bleus)
    optimal_idx, best_bleu = find_optimal(sizes, bleus)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_quality(ax1, sizes, bleus, chrfs, bleu_stds, chrf_stds, optimal_idx, best_bleu, metric_label)
    plot_marginal(ax2, sizes, gains)

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to {args.output_file}")
    print(f"Best BLEU: {best_bleu:.4f}, 95% threshold: {best_bleu*0.95:.4f}")
    if optimal_idx is not None:
        print(f"Optimal: {sizes[optimal_idx]} samples ({bleus[optimal_idx]:.4f} BLEU)")


if __name__ == "__main__":
    main()
