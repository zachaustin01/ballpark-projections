"""Local-only research code: historical store, training, backtesting.

NOTHING in the CI/deploy path (cli `export`, model.py, export.py) may import this package.
It needs the optional `research` dependency group (`uv sync --group research`).
"""
