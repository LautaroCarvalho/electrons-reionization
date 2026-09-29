"""pytest plugin for mutation runs: with MUT_MOD=<module> MUT_OLD=<text> MUT_NEW=<text> it replaces that text in the
source of igm_losses.<module> (anchor must be unique) before the tests are collected. Without MUT_MOD it does nothing."""
import importlib
import os
import pathlib


def pytest_configure(config):
    mod = os.environ.get("MUT_MOD")
    if not mod:
        return
    m = importlib.import_module("igm_losses." + mod)
    src = pathlib.Path(m.__file__).read_text(encoding="utf-8")
    old, new = os.environ["MUT_OLD"], os.environ["MUT_NEW"]
    assert src.count(old) == 1, f"mutation anchor absent or not unique in {mod}: {old!r}"
    exec(compile(src.replace(old, new), m.__file__, "exec"), m.__dict__)
