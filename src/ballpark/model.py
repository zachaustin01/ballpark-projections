"""Project where foul balls / home runs land, in seat zones, for one game.

Method (deliberately simple, all assumptions listed in README):
  * Each batter's sampled batted balls (spray angle + distance) are re-projected into
    the *game's* park. A fly ball is a HR if its distance clears the interpolated wall.
  * Balls are jittered (kernel smoothing) and binned into seat zones: a parks's
    seatmap polygons if one exists in data/seatmaps, else generic angle x depth zones.
  * Rates are per-PA, scaled by expected PA for the lineup slot and a shrunk
    pitcher factor (HR/PA and foul/PA relative to the slate's pitchers).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import statcast as sc
from . import store
from .geometry import fence_distance, load_seatmap, seat_sections, to_xy

PA_BY_SLOT = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]
SHRINK_PA = 150  # pitcher factor pseudo-sample toward 1.0
WALL_MARGIN = 8  # ft; crude allowance for wall height / non-HR balls that carry past the wall distance
JITTER = 20      # kernel copies per batted ball
RNG = np.random.default_rng(0)


def _jitter(df, rng, ang_sd=3.0, dist_sd=8.0):
    d = df.loc[df.index.repeat(JITTER)].copy()
    d["angle"] = d["angle"].to_numpy() + rng.normal(0, ang_sd, len(d))
    d["dist"] = d["dist"].to_numpy() + rng.normal(0, dist_sd, len(d))
    d["w"] = 1.0 / JITTER
    return d


def _generic_zone(angle, dist, fence, kind):
    if kind == "HR":
        lo = (np.floor((angle + 45) / 5) * 5 - 45).astype(int)
        depth = np.select([dist - fence < 20, dist - fence < 50], ["front", "mid"], "back")
        return pd.Series([f"OF {a:+d}..{a + 5:+d} deg / {d}" for a, d in zip(lo, depth)])
    side = np.where(angle < 0, "LF-side", "RF-side")
    alo = (np.floor(np.abs(angle) / 15) * 15).astype(int)
    dlo = (np.floor(dist / 50) * 50).astype(int)
    return pd.Series([f"foul {s} {a}-{a + 15} deg / {d}-{d + 50} ft"
                      for s, a, d in zip(side, alo, dlo)])


def estimate_fouls(bdf, rng=RNG):
    """Statcast has no hit coordinates for most foul balls, so landing spots are
    ESTIMATED: distance from a drag-adjusted projectile of (launch_speed, launch_angle);
    direction sampled from a pull-heavy foul-angle distribution by handedness."""
    f = bdf[bdf["is_foul"] & bdf["launch_speed"].notna() & bdf["launch_angle"].between(5, 89)].copy()
    if f.empty:
        return f
    v = f["launch_speed"].to_numpy() * 1.467  # ft/s
    f["dist"] = 0.72 * v**2 * np.sin(np.radians(2 * f["launch_angle"].to_numpy())) / 32.17
    pull = rng.random(len(f)) < 0.65
    rhb = (f["stand"] == "R").to_numpy()
    mag = np.minimum(45 + rng.exponential(30, len(f)), 175)
    behind = f["launch_angle"].to_numpy() > 60
    mag = np.where(behind, rng.uniform(140, 180, len(f)), mag)
    left = np.where(pull, rhb, ~rhb)  # RHB pulls toward left-field foul line
    f["angle"] = np.where(left, -mag, mag)
    return f


def batter_zones(bdf, park, seatmap, pitcher_hr, pitcher_foul, slot_pa, rng=RNG):
    pa = sc.n_pa(bdf)
    if pa < 20:
        return pd.DataFrame()
    out = []
    fly = bdf[(bdf["type"] == "X") & bdf["launch_angle"].between(20, 50)
              & bdf["angle"].notna() & bdf["dist"].notna()]
    fouls = estimate_fouls(bdf, rng)
    for kind, pool, factor in (("HR", fly, pitcher_hr), ("FOUL", fouls, pitcher_foul)):
        if pool.empty:
            continue
        j = _jitter(pool, rng)
        if kind == "HR":
            fence = fence_distance(j["angle"].to_numpy(), park)
            j = j[j["dist"].to_numpy() > fence + WALL_MARGIN]
            if j.empty:
                continue
            fence = fence_distance(j["angle"].to_numpy(), park)
        else:
            fence = None
        x, y = to_xy(j["angle"].to_numpy(), j["dist"].to_numpy())
        zone = seat_sections(x, y, seatmap) if seatmap else None
        if zone is None or pd.isna(zone).any():
            gen = _generic_zone(j["angle"].to_numpy(), j["dist"].to_numpy(), fence, kind).to_numpy()
            zone = gen if zone is None else np.where(pd.isna(zone), gen, zone)
        out.append(pd.DataFrame({
            "kind": kind, "zone": zone, "x": x, "y": y,
            "exp": j["w"].to_numpy() / pa * slot_pa * factor,
        }))
    return pd.concat(out) if out else pd.DataFrame()


@lru_cache(maxsize=8)
def league_rates(start, end):
    """League-wide (HR/PA, foul/PA) over the window: the baseline every pitcher is
    measured against, so one starter's factor never depends on the opposing starter."""
    df = store.load(start, end)
    pa = sc.n_pa(df)
    if not pa:
        return None
    return ((df["events"] == "home_run").sum() / pa, df["description"].isin(sc.FOUL).sum() / pa)


def pitcher_factors(pdf_by_id: dict, league=None) -> dict:
    """HR/PA and foul/PA per pitcher relative to `league` (HR/PA, foul/PA), shrunk toward 1.
    league=None falls back to the mean of the pitchers passed in (legacy behaviour)."""
    raw = {}
    for pid, df in pdf_by_id.items():
        pa = sc.n_pa(df)
        if pa:
            raw[pid] = ((df["events"] == "home_run").sum() / pa, df["is_foul"].sum() / pa, pa)
    if not raw:
        return {}
    if league:
        mhr, mfl = league
    else:
        mhr = np.mean([r[0] for r in raw.values()]) or 1
        mfl = np.mean([r[1] for r in raw.values()]) or 1
    f = {}
    for pid, (hr, fl, pa) in raw.items():
        w = pa / (pa + SHRINK_PA)
        f[pid] = (1 + w * (hr / mhr - 1), 1 + w * (fl / mfl - 1))
    return f


def project_game(game, park, date, days=365, legacy_norm=False):
    """Returns DataFrame[kind, zone, exp] summed over both lineups. Each starter's factor
    applies only to the lineup he faces and is measured against the league, not the slate."""
    start, end = sc.window(date, days)
    pitch = {}
    for s in ("away", "home"):
        p = game[s]["pitcher"]
        if p:
            pitch[p["id"]] = sc.pull_pitcher(p["id"], start, end)
    factors = pitcher_factors(pitch, None if legacy_norm else league_rates(start, end))
    rng = np.random.default_rng(game.get("game_pk", 0))  # reproducible, independent of call order
    seatmap = load_seatmap(game["venue_id"])
    frames = []
    for bat_side, pit_side in (("away", "home"), ("home", "away")):
        p = game[pit_side]["pitcher"]
        fhr, ffl = factors.get(p["id"], (1.0, 1.0)) if p else (1.0, 1.0)
        for slot, b in enumerate(game[bat_side]["lineup"][:9]):
            bdf = sc.pull_batter(b["id"], start, end)
            z = batter_zones(bdf, park, seatmap, fhr, ffl, PA_BY_SLOT[slot], rng)
            if len(z):
                z["batter"], z["team"] = b["name"], game[bat_side]["team"]
                frames.append(z)
    if not frames:
        return pd.DataFrame(columns=["kind", "zone", "x", "y", "exp"])
    return pd.concat(frames)
