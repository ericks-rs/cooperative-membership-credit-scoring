"""Composes Fig. 2 of the paper from the two source figures, ROC and calibration.

Both source figures are produced by redraw_figures.py under the same protocol
as the main experiment, logistic regression at seed 42. Here they are only
placed side by side; nothing is recomputed.

Run:      python scripts/compose_fig2.py
Writes:   figures/fig2_roc_calibration.png
"""

import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
FIGURES = Path(__file__).resolve().parents[1] / "figures"
LEFT = FIGURES / "roc_curves.png"
RIGHT = FIGURES / "calibration_curves.png"
OUT = FIGURES / "fig2_roc_calibration.png"
GAP = 60
BACKGROUND = (255, 255, 255)


def main():
    left = Image.open(LEFT).convert("RGB")
    right = Image.open(RIGHT).convert("RGB")
    width = left.width + GAP + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + GAP, (height - right.height) // 2))
    canvas.save(OUT, dpi=(600, 600))
    print(f"written to {OUT}, size {canvas.width} x {canvas.height}")


if __name__ == "__main__":
    main()
