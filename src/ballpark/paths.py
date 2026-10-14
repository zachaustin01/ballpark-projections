from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
SEATMAPS = DATA / "seatmaps"
OUTPUT = ROOT / "output"
for p in (CACHE, SEATMAPS, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)
