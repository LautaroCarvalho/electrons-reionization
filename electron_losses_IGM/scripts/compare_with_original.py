"""Side-by-side comparison of each new figure with the figure the notebook drew (C45: overlay against the source).

Reads   provenance/figures/*.json (notebook_cell of each new figure), electron_losses_IGM/reference_figures/cell<NN>_0.png
Writes  electron_losses_IGM/comparisons/compare_<name>.png  (not in figures/: these are review aids, not results)
Mapping: the notebook's cell 38 output belongs to the code of cell 39 (kernel merged, see inventory), so cell 39 → cell38_0.png;
cells 41 and 42 never produced a figure; cell 40 produced an empty one (E11).
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg          # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "electron_losses_IGM" / "reference_figures"
OUT = ROOT / "electron_losses_IGM" / "comparisons"
CELL_MAP = {39: 38}


def main():
    OUT.mkdir(exist_ok=True)
    n = 0
    for rec_path in sorted((ROOT / "provenance" / "figures").glob("*.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        cell = CELL_MAP.get(rec["notebook_cell"], rec["notebook_cell"])
        ref = REF / f"cell{cell:02d}_0.png"
        new = ROOT / rec["file"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for a, path, title in ((axes[0], ref, f"notebook, celda {rec['notebook_cell']}"), (axes[1], new, rec["file"])):
            a.axis("off"); a.set_title(title)
            if path.exists():
                a.imshow(mpimg.imread(path))
            else:
                a.text(0.5, 0.5, "sin figura en el notebook", ha="center", va="center", transform=a.transAxes)
        fig.tight_layout()
        fig.savefig(OUT / f"compare_{new.stem}.png", dpi=110)
        plt.close(fig)
        n += 1
    print(f"wrote {n} comparisons in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
