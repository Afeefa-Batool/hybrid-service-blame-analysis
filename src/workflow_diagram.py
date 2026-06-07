"""Generate a clean workflow flowchart for the replication pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_workflow(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (1.0, 8.2, "1. Chat Sessions\n(hybrid human-AI service)"),
        (5.0, 8.2, "2. Failure Filter\n(errors + delays)"),
        (9.0, 8.2, "3. LLM / NLP Labels\nblame + sentiment"),
        (1.0, 5.4, "4. Feature Table\nsession-level outcomes"),
        (5.0, 5.4, "5. First Stage\npeer IV + logit"),
        (9.0, 5.4, "6. Control Function\nendogeneity correction"),
        (3.0, 2.6, "7. Second Stage\nemotion / engagement / purchase"),
        (8.0, 2.6, "8. Results + Policy\nERA interpretation"),
    ]

    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            3.2,
            1.3,
            boxstyle="round,pad=0.08,rounding_size=0.2",
            linewidth=1.4,
            edgecolor="#1f4e79",
            facecolor="#e8f1fb",
        )
        ax.add_patch(patch)
        ax.text(x + 1.6, y + 0.65, text, ha="center", va="center", fontsize=10, color="#102a43")

    arrows = [
        ((4.2, 8.85), (5.0, 8.85)),
        ((8.2, 8.85), (9.0, 8.85)),
        ((10.6, 8.2), (10.6, 6.7)),
        ((10.6, 6.7), (2.6, 6.7)),
        ((4.2, 6.05), (5.0, 6.05)),
        ((8.2, 6.05), (9.0, 6.05)),
        ((6.6, 5.4), (6.6, 3.9)),
        ((6.6, 3.9), (4.6, 3.9)),
        ((6.2, 3.25), (8.0, 3.25)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color="#334e68",
        )
        ax.add_patch(arrow)

    ax.text(
        7.0,
        9.5,
        "Hybrid Service Blame Analysis Pipeline (Chen et al., 2025 extension)",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#102a43",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    draw_workflow(root / "assets" / "workflow_pipeline.png")
