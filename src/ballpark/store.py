"""Local league-wide Statcast store: one small parquet per game day in data/statcast/.

Savant is hit once per *day* (not once per player). After the one-time backfill a daily
run only adds the newest day; everything else is a local filter. Days whose file was
fetched within 2 days of the game date are re-pulled, since Savant fills data in late.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd
from pybaseball import statcast

from .paths import STATCAST

COLUMNS = [
    "game_date", "game_pk", "game_type", "at_bat_number", "pitch_number", "batter", "pitcher",
    "stand", "p_throws", "type", "description", "events", "launch_speed", "launch_angle",
    "launch_speed_angle", "hit_distance_sc", "hc_x", "hc_y", "estimated_woba_using_speedangle",
]
NUMERIC = ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "launch_speed",
           "launch_angle", "launch_speed_angle", "hit_distance_sc", "hc_x", "hc_y",
           "estimated_woba_using_speedangle"]
REAL_GAMES = {"R", "F", "D", "L", "W"}  # regular season + postseason; drops spring training etc.
THREADS = 4


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=COLUMNS)
    for c in NUMERIC:  # replace columns outright (pybaseball returns nullable Int64 here)
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _file(d: date):
    return STATCAST / f"{d.isoformat()}.parquet"


def _is_final(d: date) -> bool:
    f = _file(d)
    if not f.exists():
        return False
    fetched = datetime.fromtimestamp(f.stat().st_mtime).date()
    return (fetched - d).days >= 2


def _fetch_day(d: date) -> None:
    for attempt in range(4):
        try:
            df = statcast(start_dt=d.isoformat(), end_dt=d.isoformat(), verbose=False)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    if df is None or df.empty:
        df = pd.DataFrame(columns=COLUMNS)
    df = _typed(df)
    df = df[df["game_type"].isin(REAL_GAMES)] if len(df) else df
    df.to_parquet(_file(d), index=False)


def _days(start: str, end: str):
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    while d <= e:
        yield d
        d += timedelta(days=1)


def sync(start: str, end: str, verbose: bool = True) -> int:
    """Fetch any missing / not-yet-final days in [start, end]. Returns days fetched."""
    todo = [d for d in _days(start, end) if not _is_final(d)]
    if verbose and todo:
        print(f"statcast store: fetching {len(todo)} day(s) ({todo[0]} .. {todo[-1]})")
    with ThreadPoolExecutor(THREADS) as ex:
        for i, _ in enumerate(ex.map(_fetch_day, todo), 1):
            if verbose and i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    load.cache_clear()
    return len(todo)


@lru_cache(maxsize=4)
def load(start: str, end: str) -> pd.DataFrame:
    """All pitches in [start, end] (syncing first if needed)."""
    sync(start, end)
    frames = [pd.read_parquet(_file(d)) for d in _days(start, end) if _file(d).exists()]
    frames = [f for f in frames if len(f)]
    return _typed(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS))
