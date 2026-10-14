"""Research Statcast store: one parquet per game day in data/research/statcast/, with the
extra columns backtesting needs (teams, half-inning, batted-ball type). Kept separate from
the production store (data/statcast) so changing it can never affect the deploy job."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast

from ..paths import DATA

DIR = DATA / "research" / "statcast"
DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "game_date", "game_pk", "game_type", "game_year", "home_team", "away_team", "inning_topbot",
    "at_bat_number", "pitch_number", "batter", "pitcher", "stand", "p_throws", "type",
    "description", "events", "bb_type", "launch_speed", "launch_angle", "launch_speed_angle",
    "hit_distance_sc", "hc_x", "hc_y",
]
NUMERIC = ["game_pk", "game_year", "at_bat_number", "pitch_number", "batter", "pitcher",
           "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc", "hc_x", "hc_y"]
REAL_GAMES = {"R", "F", "D", "L", "W"}
# Season spans (regular season + postseason) so the backfill skips the offseason.
SEASONS = {2021: ("2021-04-01", "2021-11-03"), 2022: ("2022-04-07", "2022-11-06"),
           2023: ("2023-03-30", "2023-11-02"), 2024: ("2024-03-20", "2024-10-31"),
           2025: ("2025-03-18", "2025-11-02"), 2026: ("2026-03-25", "2026-11-05")}


def _file(d: date):
    return DIR / f"{d.isoformat()}.parquet"


def _fetch_day(d: date) -> None:
    for attempt in range(5):
        try:
            df = statcast(start_dt=d.isoformat(), end_dt=d.isoformat(), verbose=False)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    df = pd.DataFrame(columns=COLUMNS) if df is None or df.empty else df.reindex(columns=COLUMNS)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df[df["game_type"].isin(REAL_GAMES)]
    df["game_date"] = df["game_date"].astype(str)
    df.to_parquet(_file(d), index=False)


def sync(start: str, end: str, threads: int = 4) -> int:
    """Fetch missing days in [start, end]. Days within 3 days of today are always re-pulled."""
    d, e, todo = date.fromisoformat(start), date.fromisoformat(end), []
    fresh = date.today() - timedelta(days=3)
    while d <= min(e, date.today() - timedelta(days=1)):
        if not _file(d).exists() or d >= fresh:
            todo.append(d)
        d += timedelta(days=1)
    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(_fetch_day, todo), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    return len(todo)


def sync_seasons(years=None) -> None:
    for y in years or SEASONS:
        s, e = SEASONS[y]
        print(f"{y}: {s}..{e}", flush=True)
        print(f"  fetched {sync(s, e)} day(s)", flush=True)


def load(start: str, end: str) -> pd.DataFrame:
    fs = [_file(date.fromisoformat(start) + timedelta(days=i))
          for i in range((date.fromisoformat(end) - date.fromisoformat(start)).days + 1)]
    frames = [pd.read_parquet(f) for f in fs if f.exists()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
