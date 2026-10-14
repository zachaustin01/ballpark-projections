"""Batch job: compute projections and write static JSON for the web front end.

web/public/data/
  index.json                 {"dates": [...], "updated": iso}
  <date>/index.json          game summaries for the day
  <date>/<game_pk>.json      full game detail (lineups, stats, heatmap cells, zones)
"""
import json
import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import statcast as sc
from .ballparks import load_ballparks
from .geometry import DIM_KEYS
from .model import project_game
from .paths import WEB_DATA
from .schedule import get_slate

CELL_FT = 10  # heatmap grid resolution
MIN_EXP = 1e-4


def _clean(o):
    """NaN/inf -> None, numpy scalars -> python, so output is strict JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), separators=(",", ":"), allow_nan=False))


def _cells(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    g = df.assign(cx=(df.x / CELL_FT).round().astype(int) * CELL_FT,
                  cy=(df.y / CELL_FT).round().astype(int) * CELL_FT)
    g = g.groupby(["cx", "cy"], as_index=False).exp.sum()
    g = g[g.exp > MIN_EXP]
    return [[int(r.cx), int(r.cy), round(float(r.exp), 4)] for r in g.itertuples()]


def _side(g, s, date_, days, with_stats=True):
    side = g[s]
    start, end = sc.window(date_, days)
    p = side["pitcher"]
    pitcher = None
    if p:
        pitcher = {"id": p["id"], "name": p["name"]}
        if with_stats:
            pitcher["stats"] = sc.summarize(sc.pull_pitcher(p["id"], start, end), "pitcher")
    lineup = []
    for i, b in enumerate(side["lineup"], 1):
        row = {"slot": i, "id": b["id"], "name": b["name"]}
        if with_stats:
            row["stats"] = sc.summarize(sc.pull_batter(b["id"], start, end), "batter")
        lineup.append(row)
    return {"team": side["team"], "pitcher": pitcher, "lineup": lineup,
            "lineup_source": side["lineup_source"]}


def game_detail(g, parks, date_, days):
    park = parks.loc[g["venue_id"]]
    out = {
        "game_pk": g["game_pk"], "date": date_, "time": g["time"], "status": g["status"],
        "venue": {"id": int(g["venue_id"]), "name": park["name"], "capacity": park["capacity"],
                  "elevation_ft": park["elevation_ft"], "roof": park["roof"],
                  "fence": [park[k] for k in DIM_KEYS]},
        "away": _side(g, "away", date_, days), "home": _side(g, "home", date_, days),
        "lookback_days": days, "generated_at": datetime.now(timezone.utc).isoformat(),
        "projection": None,
    }
    if not (g["away"]["lineup"] and g["home"]["lineup"]):
        out["projection_note"] = "No lineup available for one or both teams yet."
        return out
    if park[list(DIM_KEYS)].isna().any():
        out["projection_note"] = "Missing fence dimensions for this park."
        return out
    res = project_game(g, park, date_, days)
    if res.empty:
        out["projection_note"] = "Not enough Statcast data."
        return out
    side_of = {g["away"]["team"]: "away", g["home"]["team"]: "home"}
    res = res.assign(side=res.team.map(side_of))
    proj = {"totals": {}, "cells": {}, "zones": {}, "batters": {}}
    for kind in ("HR", "FOUL"):
        k = res[res.kind == kind]
        proj["totals"][kind] = {s: round(float(k[k.side == s].exp.sum()), 3) for s in ("away", "home")}
        proj["cells"][kind] = {s: _cells(k[k.side == s]) for s in ("away", "home")}
        z = k.groupby("zone", as_index=False).exp.sum().sort_values("exp", ascending=False).head(12)
        proj["zones"][kind] = [{"zone": r.zone, "exp": round(float(r.exp), 3)} for r in z.itertuples()]
        b = k.groupby("batter").exp.sum()
        proj["batters"][kind] = {n: round(float(v), 3) for n, v in b.items()}
    out["projection"] = proj
    return out


def summary(d):
    p = d.get("projection")
    return {
        "game_pk": d["game_pk"], "time": d["time"], "status": d["status"],
        "venue": d["venue"]["name"],
        "away": {"team": d["away"]["team"], "pitcher": (d["away"]["pitcher"] or {}).get("name"),
                 "lineup_source": d["away"]["lineup_source"]},
        "home": {"team": d["home"]["team"], "pitcher": (d["home"]["pitcher"] or {}).get("name"),
                 "lineup_source": d["home"]["lineup_source"]},
        "has_projection": bool(p),
        "hr_total": round(sum(p["totals"]["HR"].values()), 2) if p else None,
        "foul_total": round(sum(p["totals"]["FOUL"].values()), 1) if p else None,
        "note": d.get("projection_note"),
    }


def export_date(date_, days=365, only_game=None):
    parks = load_ballparks().set_index("venue_id")
    summaries = []
    for g in get_slate(date_):
        if only_game and g["game_pk"] != only_game:
            continue
        try:
            d = game_detail(g, parks, date_, days)
        except Exception as e:  # one bad game must not sink the whole job
            print(f"  ! {g['away']['team']} @ {g['home']['team']}: {type(e).__name__}: {e}")
            continue
        _write(WEB_DATA / date_ / f"{g['game_pk']}.json", d)
        summaries.append(summary(d))
        print(f"  ok {g['away']['team']} @ {g['home']['team']}")
    summaries.sort(key=lambda s: s["time"])
    _write(WEB_DATA / date_ / "index.json",
           {"date": date_, "games": summaries, "generated_at": datetime.now(timezone.utc).isoformat()})
    update_root_index()


def update_root_index():
    dates = sorted(p.name for p in WEB_DATA.iterdir() if p.is_dir() and (p / "index.json").exists())
    _write(WEB_DATA / "index.json", {"dates": dates, "updated": datetime.now(timezone.utc).isoformat()})


def export_range(start, end, days=365, only_game=None):
    d = date.fromisoformat(start)
    while d <= date.fromisoformat(end):
        print(d.isoformat())
        export_date(d.isoformat(), days, only_game)
        d += timedelta(days=1)
