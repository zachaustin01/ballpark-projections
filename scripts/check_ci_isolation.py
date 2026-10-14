"""Fail if anything reachable from the CI/deploy path imports the research package or the
research-only dependencies. Stdlib only, so CI can run it without `uv sync --group research`."""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ballpark"
ALLOWED_LAZY = {SRC / "cli.py"}  # may import ballpark.research, but only inside a function


def imports(tree, top_level_only):
    nodes = tree.body if top_level_only else ast.walk(tree)
    for n in nodes:
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            yield ("." * n.level) + (n.module or ""), [a.name for a in n.names]


def check():
    bad = []
    for f in SRC.rglob("*.py"):
        if "research" in f.relative_to(SRC).parts:
            continue
        tree = ast.parse(f.read_text())
        for item in imports(tree, top_level_only=f in ALLOWED_LAZY):
            mod, names = (item, []) if isinstance(item, str) else item
            full = mod.lstrip(".")
            hits = [full] + [f"{full}.{n}" if full else n for n in names]
            if any(h.split(".")[0] in ("sklearn", "scipy") or h == "research" or h.startswith("research.")
                   or h.startswith("ballpark.research") for h in hits):
                bad.append(f"{f.relative_to(SRC.parents[1])}: imports {mod} {names}")
    return bad


if __name__ == "__main__":
    bad = check()
    if bad:
        print("research code leaked into the CI path:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("ok: CI path does not import research code")
