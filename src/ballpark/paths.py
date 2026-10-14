from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEATMAPS = DATA / "seatmaps"
OUTPUT = ROOT / "output"
for p in (SEATMAPS, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)

WEB_DATA = ROOT / "web" / "public" / "data"

STATCAST = DATA / "statcast"
STATCAST.mkdir(parents=True, exist_ok=True)
