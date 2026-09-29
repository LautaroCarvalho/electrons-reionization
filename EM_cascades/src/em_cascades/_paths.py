"""Import path setup shared by scripts and tests: project root, igm_losses on sys.path, no bytecode written.

The task forbids creating files inside electron_losses_IGM/, and importing igm_losses would otherwise write
__pycache__ there. Importing this module first sets sys.dont_write_bytecode before igm_losses is imported.
"""

import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in (ROOT / "electron_losses_IGM" / "src", ROOT / "EM_cascades" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
