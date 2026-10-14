"""Turn raw pitches into the tables the models and the backtest use.

All times are integer day numbers (`dn`, days since 2000-01-01) so rolling windows are a
searchsorted away. Everything here is a pure function of the historical store."""
import numpy as np
import pandas as pd

from ..geometry import ANGLES, DIM_KEYS, spray
from . import data
from .games import load_games, load_venues

EPOCH = pd.Timestamp("2000-01-01")
FOUL = {"foul", "foul_tip", "foul_bunt"}
SLOTS = 9


def day_number(s) -> np.ndarray:
    return (pd.to_datetime(s) - EPOCH).dt.days.to_numpy()


class Venues:
    """Fence profile (7 points, gaps interpolated like geometry.fence_profile), elevation,
    roof flag, indexed by a dense integer `vi`."""

    def __init__(self):
        v = load_venues().drop_duplicates("venue_id").reset_index(drop=True)
        self.df = v
        self.ids = v.venue_id.to_numpy()
        self.idx = {int(i): k for k, i in enumerate(self.ids)}
        prof = []
        for _, r in v.iterrows():
            vals = np.array([r[k] if r[k] is not None else np.nan for k in DIM_KEYS], float)
            ok = ~np.isnan(vals)
            prof.append(np.interp(ANGLES, ANGLES[ok], vals[ok]) if ok.sum() >= 2 else np.full(7, np.nan))
        self.fence = np.array(prof)                                  # [n_venues, 7]
        self.elev = pd.to_numeric(v.elevation_ft, errors="coerce").fillna(500).to_numpy(float)
        self.roof = v.roof.fillna("Open").str.lower().isin(["dome", "retractable"]).to_numpy(float)

    def fence_at(self, vi, angle):
        """Wall distance at spray angle(s) for venue index(es) `vi` (scalar or array)."""
        a = np.asarray(angle, float)
        if np.ndim(vi) == 0:
            return np.interp(a, ANGLES, self.fence[vi])
        out = np.empty(len(a))
        for k in np.unique(vi):
            m = vi == k
            out[m] = np.interp(a[m], ANGLES, self.fence[k])
        return out

    def vi(self, venue_ids):
        return pd.Series(venue_ids).map(self.idx).to_numpy()


def build(start: str, end: str, venues: Venues):
    """Returns dict(pa, balls, lineups) for games in [start, end]."""
    px = data.load(start, end)
    games = load_games().set_index("game_pk")
    px = px[px.game_pk.isin(games.index)].copy()
    px["is_foul"] = px.description.isin(FOUL)
    px["hr"] = px.events.eq("home_run")
    px["dn"] = day_number(px.game_date)
    px["vi"] = venues.vi(px.game_pk.map(games.venue_id))
    px["bat_side"] = np.where(px.inning_topbot.eq("Top"), "away", "home")

    pa = (px.sort_values(["game_pk", "at_bat_number", "pitch_number"])
          .groupby(["game_pk", "at_bat_number"], sort=False)
          .agg(dn=("dn", "first"), vi=("vi", "first"), side=("bat_side", "first"),
               batter=("batter", "first"), pitcher=("pitcher", "first"),
               stand=("stand", "first"), p_throws=("p_throws", "first"),
               hr=("hr", "max"), fouls=("is_foul", "sum"), pitches=("pitch_number", "count"),
               year=("game_year", "first"))
          .reset_index())
    pa["hr"] = pa.hr.astype(int)
    pa["fouls"] = pa.fouls.astype(int)
    for c in ("batter", "pitcher", "year", "game_pk"):
        pa[c] = pa[c].astype("int64")

    # ---- batted balls (fair balls in play with tracking) ----
    b = px[(px.type == "X") & px.launch_speed.notna() & px.launch_angle.notna() & px.hc_x.notna()].copy()
    _, _, ang, hdist = spray(b.hc_x, b.hc_y)
    b["angle"] = ang
    b["dist"] = b.hit_distance_sc.fillna(pd.Series(hdist, index=b.index))
    b["fence"] = venues.fence_at(b.vi.to_numpy().astype(int), ang)
    balls = b[["game_pk", "at_bat_number", "dn", "vi", "batter", "pitcher", "stand", "p_throws",
               "launch_speed", "launch_angle", "angle", "dist", "fence", "hr", "game_year"]].reset_index(drop=True)
    balls["hr"] = balls.hr.astype(int)
    balls = balls[balls.vi.notna()].reset_index(drop=True)
    balls["vi"] = balls.vi.astype(int)
    for c in ("batter", "pitcher", "game_pk"):
        balls[c] = balls[c].astype("int64")

    # ---- lineups: first nine distinct batters per team-game, ordered by first PA ----
    first = pa.drop_duplicates(["game_pk", "side", "batter"]).sort_values(["game_pk", "side", "at_bat_number"])
    first["slot"] = first.groupby(["game_pk", "side"]).cumcount() + 1
    lineups = first[first.slot <= SLOTS][["game_pk", "side", "batter", "slot", "dn", "vi", "stand", "year"]]
    # opposing starter = pitcher of that team's first PA
    sp = (pa.sort_values(["game_pk", "at_bat_number"]).drop_duplicates(["game_pk", "side"])
          [["game_pk", "side", "pitcher", "p_throws"]].rename(columns={"pitcher": "sp", "p_throws": "sp_hand"}))
    lineups = lineups.merge(sp, on=["game_pk", "side"], how="left")
    # truth for lineup members
    truth = pa.groupby(["game_pk", "side", "batter"]).agg(pa=("hr", "size"), hr=("hr", "sum"),
                                                          fouls=("fouls", "sum")).reset_index()
    lineups = lineups.merge(truth, on=["game_pk", "side", "batter"], how="left")
    home = lineups.side.eq("home")
    lineups["home"] = home.astype(int)
    lineups["temp"] = lineups.game_pk.map(games.temp)
    return {"pa": pa, "balls": balls, "lineups": lineups.reset_index(drop=True)}


def hr_truth(balls: pd.DataFrame, venues: Venues) -> pd.DataFrame:
    """Actual HRs with landing spot (angle from hit coordinates, depth from hit distance)."""
    h = balls[balls.hr == 1].copy()
    h["excess"] = h.dist - h.fence
    return h
