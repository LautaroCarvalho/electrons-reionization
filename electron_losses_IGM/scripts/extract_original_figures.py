"""Extract the figures embedded in the original notebook, for side-by-side comparison.

Purpose
    The original notebook (Perdidas_de_energia-Z_10.ipynb, by L. Carvalho) stores every figure it drew
    as a base64 PNG in its cell outputs. This script writes each one to
    electron_losses_IGM/reference_figures/cell<NN>_<k>.png and a JSON index with the cell number,
    the execution count, the image size and the sha256 of each PNG.

Reads
    provenance/original/Perdidas_de_energia-Z_10.ipynb   (copy of the original; see provenance/MANIFEST.txt)
    It never opens or writes anything inside contenido_inicial_generado_por_Carvalho/.

Writes
    electron_losses_IGM/reference_figures/cell<NN>_<k>.png
    electron_losses_IGM/reference_figures/index.json

Physics
    None. This is a file-format operation: nothing is computed.

Run
    python3 electron_losses_IGM/scripts/extract_original_figures.py      (from the project root)
"""

import base64
import hashlib
import json
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "provenance" / "original" / "Perdidas_de_energia-Z_10.ipynb"
OUT_DIR = ROOT / "electron_losses_IGM" / "reference_figures"


def png_size(data):
    """Width and height in pixels, read from the PNG IHDR chunk (bytes 16-24)."""
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def extract(notebook=NOTEBOOK, out_dir=OUT_DIR):
    """Write every embedded PNG to out_dir and return the index (list of dicts)."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for cell_no, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        k = 0
        for output in cell.get("outputs", []):
            b64 = output.get("data", {}).get("image/png")
            if b64 is None:
                continue
            data = base64.b64decode("".join(b64) if isinstance(b64, list) else b64)
            name = f"cell{cell_no:02d}_{k}.png"
            (out_dir / name).write_bytes(data)
            width, height = png_size(data)
            index.append({"file": name, "cell": cell_no, "execution_count": cell.get("execution_count"),
                          "width_px": width, "height_px": height,
                          "sha256": hashlib.sha256(data).hexdigest()})
            k += 1
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    idx = extract()
    print(f"{len(idx)} figures extracted to {OUT_DIR.relative_to(ROOT)}")
    for entry in idx:
        print(f"  {entry['file']:<16} cell {entry['cell']:>2}  exec {str(entry['execution_count']):>4}  "
              f"{entry['width_px']}x{entry['height_px']}")
