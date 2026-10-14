"""Games on a date with probable pitchers and (if posted) batting orders."""
from datetime import date, timedelta
from functools import lru_cache

from . import mlb_api

LOOKBACK_DAYS = 60


@lru_cache(maxsize=None)
def pitch_hand(pid: int) -> str | None:
    return mlb_api.get(f"people/{pid}")["people"][0].get("pitchHand", {}).get("code")


@lru_cache(maxsize=None)
def _boxscore(game_pk: int) -> dict:
    return mlb_api.get(f"game/{game_pk}/boxscore")


def projected_lineup(team_id: int, opp_pitcher_id: int | None, game_date: str):
    """Team's most recent lineup against a starter with the same throwing hand as
    today's opposing probable pitcher (any hand if that pitcher is unknown).
    Returns (lineup, source note) or ([], None)."""
    want = pitch_hand(opp_pitcher_id) if opp_pitcher_id else None
    end = date.fromisoformat(game_date) - timedelta(days=1)
    games = mlb_api.get(
        "schedule", sportId=1, teamId=team_id,
        startDate=(end - timedelta(days=LOOKBACK_DAYS)).isoformat(), endDate=end.isoformat(),
    )
    finals = [g for d in games["dates"] for g in d["games"]
              if g["status"]["abstractGameState"] == "Final"]
    for g in sorted(finals, key=lambda g: g["gameDate"], reverse=True):
        side = "home" if g["teams"]["home"]["team"]["id"] == team_id else "away"
        opp = "away" if side == "home" else "home"
        box = _boxscore(g["gamePk"])["teams"]
        if not box[opp]["pitchers"]:
            continue
        hand = pitch_hand(box[opp]["pitchers"][0])
        if want and hand != want:
            continue
        order = box[side]["battingOrder"]
        if len(order) < 9:
            continue
        lineup = [{"id": i, "name": box[side]["players"][f"ID{i}"]["person"]["fullName"]}
                  for i in order[:9]]
        note = f"projected from {g['officialDate']} vs {hand}HP"
        return lineup, note
    return [], None


def get_slate(date: str, project_lineups: bool = True) -> list[dict]:
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
                    "team_id": team["team"]["id"],
                    "lineup_source": "posted",
                    "pitcher": {"id": pp["id"], "name": pp["fullName"]} if pp else None,
                    "lineup": [
                        {"id": p["id"], "name": p["fullName"]}
                        for p in lineups.get(f"{s}Players", [])
                    ],
                }
            if project_lineups and g["status"]["abstractGameState"] == "Preview":
                for s, o in (("away", "home"), ("home", "away")):
                    if not side[s]["lineup"]:
                        opp_p = side[o]["pitcher"]
                        lu, note = projected_lineup(
                            side[s]["team_id"], opp_p["id"] if opp_p else None, d["date"])
                        side[s]["lineup"], side[s]["lineup_source"] = lu, note or "none"
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
