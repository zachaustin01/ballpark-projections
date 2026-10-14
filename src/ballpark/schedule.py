"""Games on a date with probable pitchers and (if posted) batting orders."""
from . import mlb_api


def get_slate(date: str) -> list[dict]:
    dates = mlb_api.get(
        "schedule", sportId=1, date=date, hydrate="probablePitcher,lineups,venue"
    )["dates"]
    games = []
    for d in dates:
        for g in d["games"]:
            lineups = g.get("lineups", {})
            side = {}
            for s in ("away", "home"):
                team = g["teams"][s]
                pp = team.get("probablePitcher")
                side[s] = {
                    "team": team["team"]["name"],
                    "pitcher": {"id": pp["id"], "name": pp["fullName"]} if pp else None,
                    "lineup": [
                        {"id": p["id"], "name": p["fullName"]}
                        for p in lineups.get(f"{s}Players", [])
                    ],
                }
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "time": g["gameDate"],
                    "status": g["status"]["detailedState"],
                    "venue_id": g["venue"]["id"],
                    "venue": g["venue"]["name"],
                    **{f"{s}": side[s] for s in side},
                }
            )
    return games
