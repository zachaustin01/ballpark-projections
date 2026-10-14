import argparse
from datetime import date, timedelta

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
            print(f"\n{side['team']}" + (f"  ({side['lineup_source']})" if side["lineup"] and side["lineup_source"] != "posted" else "" if side["lineup"] else "  (no lineup)"))
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


def cmd_sync(a):
    from . import store
    end = a.through or (date.today() - timedelta(days=1)).isoformat()
    start = (date.fromisoformat(end) - timedelta(days=a.days)).isoformat()
    n = store.sync(start, end)
    print(f"done ({n} days fetched)")


def cmd_export(a):
    from .export import export_range
    today = date.today()
    start = a.date or today.isoformat()
    end = a.through or (today + timedelta(days=1)).isoformat() if not a.date else (a.through or a.date)
    export_range(start, end, a.days, a.game)


def cmd_research(a):
    """Local-only research tooling. Imported lazily: needs `uv sync --group research` and
    must never be reachable from the CI/export path."""
    import json
    from dataclasses import fields
    from .research import backtest, data
    if a.rcmd == "sync":
        data.sync_seasons([int(y) for y in a.seasons] if a.seasons else None)
        from .research import games
        games.load_games(refresh=True)
        games.load_venues(refresh=True)
    elif a.rcmd == "ablate":
        from .research import ablations
        df = ablations.run(a.test_year)
        print(df.round(4).to_string(index=False))
        df.to_csv(OUTPUT / "backtests" / f"classifier_ablation_{a.test_year}.csv", index=False)
    elif a.rcmd == "report":
        from .research import report
        print(report.render(a.name))
    elif a.rcmd == "backtest":
        from .research.engine import Params
        params = Params(**json.loads(open(a.params).read())) if a.params else None
        backtest.run(a.name, a.test_start, a.test_end, a.train_through, a.every, params,
                     a.tune, not a.no_v0, a.with_dist)


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
    p = sub.add_parser("sync", help="update the local league-wide Statcast store")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--through", help="last day to fetch (default: yesterday)")
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("export", help="write static JSON for the web front end")
    p.add_argument("--date", help="start date (default: today)")
    p.add_argument("--through", help="end date inclusive (default: --date, or tomorrow if no --date)")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--game", type=int, help="only this gamePk (testing)")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("research", help="LOCAL ONLY: historical store + backtesting (never run in CI)")
    rs = p.add_subparsers(dest="rcmd", required=True)
    q = rs.add_parser("sync", help="backfill the research Statcast store + game/venue tables")
    q.add_argument("--seasons", nargs="*")
    q = rs.add_parser("ablate", help="ball-level HR classifier ablations (distance, venue id, temperature)")
    q.add_argument("--test-year", type=int, default=2024)
    q = rs.add_parser("report", help="render report.md/PNGs for a finished backtest")
    q.add_argument("name")
    q = rs.add_parser("backtest", help="walk-forward backtest of v0 and the candidate model")
    q.add_argument("--name", required=True)
    q.add_argument("--test-start", required=True)
    q.add_argument("--test-end", required=True)
    q.add_argument("--train-through", type=int, required=True, help="HR classifier uses seasons <= this")
    q.add_argument("--every", type=int, default=7, help="run slow v0 on every Nth day")
    q.add_argument("--tune", action="store_true", help="fit shrinkage/kernel params on this window")
    q.add_argument("--params", help="params.json from a previous --tune run (frozen)")
    q.add_argument("--no-v0", action="store_true")
    q.add_argument("--with-dist", action="store_true", help="classifier may use hit distance")
    p.set_defaults(fn=cmd_research)
    a = ap.parse_args()
    a.fn(a)
