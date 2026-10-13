"""Restyle the preserved comparison report without running models or evaluation.

Run this presentation-only helper with Python and Matplotlib. It reads the
existing final comparison JSON and writes only its PNG and SVG figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "initial": "#818681",
    "selected": "#27664B",
    "evidence": "#BAA67A",
    "text": "#28332D",
    "grid": "#D7DED8",
    "panel": "#FAFBF9",
    "error": "#A24C44",
    "error_background": "#FCF3F0",
}


def render(report: Path, output_directory: Path) -> dict:
    result = json.loads(report.read_text(encoding="utf-8"))
    original = [
        result["original_qa"]["answerable_exact_match"],
        result["original_qa"]["answerable_character_f1"],
        result["original_retrieval"]["recall_at_5"],
    ]
    selected = [
        result["selected_qa"]["answerable_exact_match"],
        result["selected_qa"]["answerable_character_f1"],
        result["selected_retrieval"]["recall_at_5"],
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "axes.edgecolor": PALETTE["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "svg.hashsalt": "korean-rag-presentation-v1",
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    positions, width = np.arange(3), .34
    for offset, values, label, color in (
        (-width / 2, original, "Initial model", PALETTE["initial"]),
        (width / 2, selected, "Final model", PALETTE["selected"]),
    ):
        bars = axes[0].bar(positions + offset, np.array(values) * 100, width,
                           label=label, color=color, zorder=3)
        axes[0].bar_label(bars, labels=[f"{value * 100:.2f}%" for value in values],
                          padding=5, fontsize=11)
    axes[0].set_xticks(positions, ["Exact Match", "Character F1", "Recall@5"])
    axes[0].set_title("Answerable project holdout (n = 501)", pad=16, fontsize=13)
    axes[0].set_ylabel("Percent (%)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.095), ncol=2,
                   frameon=True, facecolor="white", framealpha=1,
                   edgecolor=PALETTE["evidence"], fontsize=11,
                   borderpad=.55, columnspacing=1.8)

    rates = [result[key]["no_answer_abstention_rate"] * 100
             for key in ("original_qa", "selected_qa")]
    labels = [f"{result[key]['no_answer_abstained_count']}/{result[key]['no_answer_count']}\n"
              f"{result[key]['no_answer_abstention_rate'] * 100:.2f}%"
              for key in ("original_qa", "selected_qa")]
    bars = axes[1].bar([0, 1], rates, width=.55,
                       color=[PALETTE["initial"], PALETTE["selected"]], zorder=3)
    axes[1].bar_label(bars, labels=labels, padding=5, fontsize=11)
    axes[1].set_xticks([0, 1], ["Initial", "Final"])
    axes[1].set_title("Synthetic no-answer (n = 80)", pad=16, fontsize=13)
    axes[1].set_ylabel("Abstention (%)")
    original_qa, selected_qa = result["original_qa"], result["selected_qa"]
    annotation = ("False answers\n"
                  f"{original_qa['no_answer_false_answer_count']}/{original_qa['no_answer_count']}"
                  "  →  "
                  f"{selected_qa['no_answer_false_answer_count']}/{selected_qa['no_answer_count']}")
    axes[1].text(.5, .48, annotation, transform=axes[1].transAxes,
                 ha="center", fontsize=12, color=PALETTE["error"],
                 bbox={"boxstyle": "round,pad=.6", "facecolor": PALETTE["error_background"],
                       "edgecolor": PALETTE["error"], "linewidth": 1.1})
    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.set_ylim(0, 110)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color=PALETTE["grid"], alpha=.65, linewidth=.7, zorder=0)
        ax.tick_params(axis="both", length=4, width=.8)
        for spine in ax.spines.values():
            spine.set_linewidth(.9)
    fig.suptitle("Korean NLP RAG: frozen final comparison", fontsize=18,
                 fontweight="bold", y=.965)
    fig.text(.5, .025,
             "Paired Windows GPU measurement. Public KorQuAD project holdout; controlled synthetic cases are separate.",
             ha="center", fontsize=10, color=PALETTE["text"])
    fig.subplots_adjust(left=.075, right=.982, bottom=.20, top=.79, wspace=.25)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [output_directory / "final-comparison.png", output_directory / "final-comparison.svg"]
    fig.savefig(paths[0], dpi=160, facecolor="white")
    fig.savefig(paths[1], facecolor="white", metadata={"Date": None})
    plt.close(fig)
    return {"figure_paths": [str(path) for path in paths], "palette": PALETTE,
            "models_or_evaluation_run": False}


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=repository / "reports/final/final-comparison.json")
    parser.add_argument("--output-directory", type=Path, default=repository / "reports")
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.output_directory)))


if __name__ == "__main__":
    main()
