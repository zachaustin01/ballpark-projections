"""Pull every MLB ballpark (location + fence dimensions) into data/ballparks.json."""
import json

import pandas as pd

from . import mlb_api
from .paths import DATA

FILE = DATA / "ballparks.json"


def fetch_ballparks() -> pd.DataFrame:
    teams = mlb_api.get("teams", sportId=1, hydrate="venue")["teams"]
    venue_team = {t["venue"]["id"]: t["name"] for t in teams}
    venues = mlb_api.get(
        "venues", venueIds=",".join(map(str, venue_team)), hydrate="location,fieldInfo"
    )["venues"]
    rows = []
    for v in venues:
        loc, fi = v.get("location", {}), v.get("fieldInfo", {})
        coords = loc.get("defaultCoordinates", {})
        rows.append(
            {
                "venue_id": v["id"],
                "name": v["name"],
                "team": venue_team[v["id"]],
                "city": loc.get("city"),
                "state": loc.get("stateAbbrev"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "azimuth": loc.get("azimuthAngle"),
                "elevation_ft": loc.get("elevation"),
                "capacity": fi.get("capacity"),
                "roof": fi.get("roofType"),
                "turf": fi.get("turfType"),
                **{k: fi.get(k) for k in ("leftLine", "left", "leftCenter", "center",
                                          "rightCenter", "right", "rightLine")},
            }
        )
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def save_ballparks() -> pd.DataFrame:
    df = fetch_ballparks()
    FILE.write_text(df.to_json(orient="records", indent=2))
    df.to_csv(DATA / "ballparks.csv", index=False)
    return df


def load_ballparks(refresh: bool = False) -> pd.DataFrame:
    if refresh or not FILE.exists():
        return save_ballparks()
    return pd.DataFrame(json.loads(FILE.read_text()))
