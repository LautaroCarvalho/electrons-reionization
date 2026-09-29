"""pytest plugin for EM_cascades mutation runs: MUT_MOD=<em_cascades module> MUT_OLD/MUT_NEW replace source text
(anchor must be unique) before collection. Without MUT_MOD it does nothing. Bytecode is never written."""
import importlib
import os
import pathlib
import sys

sys.dont_write_bytecode = True


def pytest_configure(config):
    mod = os.environ.get("MUT_MOD")
    if not mod:
        return
    import em_cascades._paths  # noqa: F401
    m = importlib.import_module("em_cascades." + mod)
    src = pathlib.Path(m.__file__).read_text(encoding="utf-8")
    old, new = os.environ["MUT_OLD"], os.environ["MUT_NEW"]
    assert src.count(old) == 1, f"mutation anchor absent or not unique in {mod}: {old!r}"
    exec(compile(src.replace(old, new), m.__file__, "exec"), m.__dict__)
