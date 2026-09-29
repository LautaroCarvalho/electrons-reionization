"""pytest setup for EM_cascades: no bytecode (nothing may be written inside electron_losses_IGM/), paths."""
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import em_cascades._paths  # noqa: E402,F401
