"""Write the printable calibration card: docs/wound/calibration_card.png (300 dpi) and .pdf (A4, 4 cards, real size).

    python scripts/make_calibration_card.py

Print the PDF at 100 % ("actual size", no "fit to page"), then check the 50 mm ruler with a real ruler.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surgiguard_d1.wound.calibration import DEFAULT_CARD  # noqa: E402
from surgiguard_d1.wound.card_render import render_card  # noqa: E402

DPI = 300
PX_PER_MM = DPI / 25.4
A4_MM = (210, 297)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    card = cv2.cvtColor(render_card(DEFAULT_CARD, PX_PER_MM), cv2.COLOR_BGR2RGB)
    Image.fromarray(card).save(out_dir / "calibration_card.png", dpi=(DPI, DPI))
    page = np.full((int(A4_MM[1] * PX_PER_MM), int(A4_MM[0] * PX_PER_MM), 3), 255, np.uint8)
    h, w = card.shape[:2]
    gap = int(10 * PX_PER_MM)
    for row in range(2):
        for col in range(2):
            y, x = gap + row * (h + gap), gap + col * (w + gap)
            page[y:y + h, x:x + w] = card
    cv2.putText(page, "Print at 100% (actual size). Check: the ruler must measure exactly 50 mm.",
                (gap, int(2 * (h + gap) + gap + 8 * PX_PER_MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(page, "Place the card flat on the skin next to the wound, fully visible, phone parallel to the skin.",
                (gap, int(2 * (h + gap) + gap + 16 * PX_PER_MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
    Image.fromarray(page).save(out_dir / "calibration_card.pdf", resolution=DPI)
    print("wrote", out_dir / "calibration_card.png", "and", out_dir / "calibration_card.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "docs" / "wound")
