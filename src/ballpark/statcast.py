"""Cached Statcast pulls (via pybaseball) and per-player summary metrics."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .geometry import spray
from . import store

FOUL = {"foul", "foul_tip", "foul_bunt"}


def window(game_date: str, days: int) -> tuple[str, str]:
    d = date.fromisoformat(game_date)
    return (d - timedelta(days=days)).isoformat(), (d - timedelta(days=1)).isoformat()


def pull_batter(pid, start, end):
    df = store.load(start, end)
    return prep(df[df["batter"] == pid])


def pull_pitcher(pid, start, end):
    df = store.load(start, end)
    return prep(df[df["pitcher"] == pid])


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add field x/y (ft from home), spray angle (neg = LF), and landing distance."""
    if df.empty:
        return df
    df = df.copy()
    df["x"], df["y"], df["angle"], d = spray(df["hc_x"], df["hc_y"])
    df["dist"] = df["hit_distance_sc"].fillna(pd.Series(d, index=df.index))
    df["is_foul"] = df["description"].isin(FOUL)
    return df


def n_pa(df: pd.DataFrame) -> int:
    return int(df.drop_duplicates(["game_pk", "at_bat_number"]).shape[0]) if len(df) else 0


def summarize(df: pd.DataFrame, side: str) -> dict:
    """Headline metrics for the slate view. side = 'batter' | 'pitcher'."""
    pa = n_pa(df)
    if not pa:
        return {"PA": 0}
    bbe = df[(df["type"] == "X") & df["launch_speed"].notna()]
    out = {
        "PA": pa,
        "BBE": len(bbe),
        "EV": round(bbe["launch_speed"].mean(), 1),
        "maxEV": round(bbe["launch_speed"].max(), 1),
        "LA": round(bbe["launch_angle"].mean(), 1),
        "Brl%": round(100 * (bbe["launch_speed_angle"] == 6).mean(), 1),
        "HH%": round(100 * (bbe["launch_speed"] >= 95).mean(), 1),
        "HR": int((df["events"] == "home_run").sum()),
        "Foul/PA": round(df["is_foul"].sum() / pa, 2),
    }
    if "xwoba" in df or "estimated_woba_using_speedangle" in df:
        out["xwOBAcon"] = round(bbe["estimated_woba_using_speedangle"].mean(), 3)
    return out
