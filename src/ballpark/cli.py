import argparse
from datetime import date

import pandas as pd

from . import statcast as sc
from .ballparks import load_ballparks
from .model import project_game
from .paths import OUTPUT
from .schedule import get_slate

pd.set_option("display.width", 200, "display.max_columns", 30)


def cmd_parks(a):
    df = load_ballparks(refresh=True)
    print(df[["venue_id", "name", "team", "capacity", "leftLine", "center", "rightLine"]]
          .to_string(index=False))
    print(f"\n{len(df)} parks saved to data/ballparks.json / .csv")


def cmd_slate(a):
    games = get_slate(a.date)
    start, end = sc.window(a.date, a.days)
    for g in games:
        print(f"\n== {g['away']['team']} @ {g['home']['team']}  ({g['venue']}, {g['status']})")
        for s in ("away", "home"):
            side, rows = g[s], []
            p = side["pitcher"]
            if p and not a.no_stats:
                rows.append({"role": "SP", "name": p["name"], **sc.summarize(sc.pull_pitcher(p["id"], start, end), "pitcher")})
            elif p:
                rows.append({"role": "SP", "name": p["name"]})
            for i, b in enumerate(side["lineup"], 1):
                st = {} if a.no_stats else sc.summarize(sc.pull_batter(b["id"], start, end), "batter")
                rows.append({"role": str(i), "name": b["name"], **st})
            print(f"\n{side['team']}" + ("" if side["lineup"] else "  (lineup not posted)"))
            if rows:
                print(pd.DataFrame(rows).to_string(index=False))


def cmd_project(a):
    parks = load_ballparks().set_index("venue_id")
    for g in get_slate(a.date):
        if a.game and g["game_pk"] != a.game:
            continue
        if not (g["away"]["lineup"] and g["home"]["lineup"]):
            print(f"skip {g['away']['team']} @ {g['home']['team']}: lineups not posted")
            continue
        park = parks.loc[g["venue_id"]]
        if park[["leftLine", "center", "rightLine"]].isna().any():
            print(f"skip {g['venue']}: missing fence dimensions")
            continue
        res = project_game(g, park, a.date, a.days)
        name = f"{a.date}_{g['game_pk']}"
        res.to_csv(OUTPUT / f"{name}_by_batter.csv", index=False)
        zones = res.groupby(["kind", "zone"], as_index=False)["exp"].sum()
        zones.to_csv(OUTPUT / f"{name}_zones.csv", index=False)
        print(f"\n== {g['away']['team']} @ {g['home']['team']} ({g['venue']})")
        for kind in ("HR", "FOUL"):
            z = zones[zones.kind == kind].sort_values("exp", ascending=False)
            print(f"\nTop {kind} zones (expected count this game; total {z.exp.sum():.2f}):")
            print(z.head(a.top).assign(exp=lambda d: d.exp.round(3)).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(prog="ballpark")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("parks", help="pull all MLB ballparks"); p.set_defaults(fn=cmd_parks)
    for name, fn in (("slate", cmd_slate), ("project", cmd_project)):
        p = sub.add_parser(name)
        p.add_argument("--date", default=date.today().isoformat())
        p.add_argument("--days", type=int, default=365, help="Statcast lookback window")
        p.set_defaults(fn=fn)
        if name == "slate":
            p.add_argument("--no-stats", action="store_true")
        else:
            p.add_argument("--game", type=int, help="gamePk")
            p.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    a.fn(a)
