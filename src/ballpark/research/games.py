"""game_pk -> venue (+ first-pitch weather) for historical games, and fence data for every
venue that appears. Cached under data/research/. The production ballparks.json only lists
each team's *current* park, so one-off/temporary venues are fetched here."""
import json
from datetime import date, timedelta

import pandas as pd

from .. import mlb_api
from ..ballparks import load_ballparks
from ..paths import DATA
from .data import SEASONS

GAMES = DATA / "research" / "games.parquet"
VENUES = DATA / "research" / "venues.json"
FENCE_KEYS = ["leftLine", "left", "leftCenter", "center", "rightCenter", "right", "rightLine"]


def fetch_games(years=None) -> pd.DataFrame:
    rows = []
    for y in years or SEASONS:
        d, end = (date.fromisoformat(x) for x in SEASONS[y])
        while d <= end:
            e = min(d + timedelta(days=30), end)
            res = mlb_api.get("schedule", sportId=1, startDate=d.isoformat(), endDate=e.isoformat(),
                              hydrate="weather,venue", gameType="R,F,D,L,W")
            for dd in res["dates"]:
                for g in dd["games"]:
                    if g["status"]["abstractGameState"] != "Final":
                        continue
                    w = g.get("weather", {})
                    rows.append({"game_pk": g["gamePk"], "date": dd["date"], "venue_id": g["venue"]["id"],
                                 "temp": pd.to_numeric(w.get("temp"), errors="coerce"),
                                 "wind": w.get("wind"), "condition": w.get("condition")})
            d = e + timedelta(days=1)
    df = pd.DataFrame(rows).drop_duplicates("game_pk")
    GAMES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GAMES, index=False)
    return df


def fetch_venues(venue_ids) -> pd.DataFrame:
    res = mlb_api.get("venues", venueIds=",".join(map(str, sorted(venue_ids))), hydrate="location,fieldInfo")
    rows = []
    for v in res["venues"]:
        loc, fi = v.get("location", {}), v.get("fieldInfo", {})
        rows.append({"venue_id": v["id"], "name": v["name"], "elevation_ft": loc.get("elevation"),
                     "roof": fi.get("roofType"), **{k: fi.get(k) for k in FENCE_KEYS}})
    df = pd.DataFrame(rows)
    # prefer the curated production table where it has the venue
    cur = load_ballparks().set_index("venue_id")
    for i, r in df.iterrows():
        if r.venue_id in cur.index:
            for k in FENCE_KEYS + ["elevation_ft", "roof"]:
                df.at[i, k] = cur.at[r.venue_id, k]
    VENUES.write_text(df.to_json(orient="records", indent=1))
    return df


def load_games(refresh=False) -> pd.DataFrame:
    return fetch_games() if refresh or not GAMES.exists() else pd.read_parquet(GAMES)


def load_venues(game_df=None, refresh=False) -> pd.DataFrame:
    if refresh or not VENUES.exists():
        game_df = load_games() if game_df is None else game_df
        return fetch_venues(set(game_df.venue_id))
    return pd.DataFrame(json.loads(VENUES.read_text()))
