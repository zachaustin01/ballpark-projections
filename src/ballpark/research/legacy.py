"""Run the production (v0) projector over historical games so it can be scored by the same
backtest. Patches store.load to read the research store, so nothing touches data/statcast."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import model, store
from . import data
from .engine import _cell_index


@lru_cache(maxsize=3)
def _load(start, end):
    return data.load(start, end)


def run(eng, R: pd.DataFrame, legacy_norm: bool, log=print) -> pd.DataFrame:
    """Returns lineup-row-aligned frame with lam_hr, lam_foul and v0 HR cells."""
    orig = store.load
    store.load = _load
    out = np.zeros((len(R), 2))
    cells = np.zeros((len(R), 90))
    v = eng.venues
    try:
        key = {(g, b): i for i, (g, b) in enumerate(zip(R.game_pk, R.batter))}
        days = sorted(R.dn.unique())
        for n, dn in enumerate(days):
            date = (pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(dn))).date().isoformat()
            for gpk, g in R[R.dn == dn].groupby("game_pk"):
                if len(g.side.unique()) < 2:
                    continue
                vi = int(g.vi.iloc[0])
                game = {"game_pk": int(gpk), "venue_id": int(v.ids[vi])}
                for side in ("away", "home"):
                    gs = g[g.side == side].sort_values("slot")
                    game[side] = {"team": side, "pitcher": None,
                                  "lineup": [{"id": int(b), "name": str(int(b))} for b in gs.batter],
                                  "_sp": int(gs.sp.iloc[0])}
                for side, other in (("away", "home"), ("home", "away")):
                    # the starter a side's *own* pitcher slot holds is the one the other side's lineup faces
                    game[side]["pitcher"] = {"id": game[other]["_sp"], "name": "sp"}
                park = v.df.iloc[vi]
                if park[["leftLine", "center", "rightLine"]].isna().any():
                    continue
                res = model.project_game(game, park, date, 365, legacy_norm=legacy_norm)
                if res.empty:
                    continue
                res["batter"] = res.batter.astype(int)
                res["side"] = res.team
                for (kind, side, b), x in res.groupby(["kind", "side", "batter"]):
                    i = key.get((gpk, b))
                    if i is None:
                        continue
                    out[i, 0 if kind == "HR" else 1] += x.exp.sum()
                    if kind == "HR":
                        ang = np.degrees(np.arctan2(x.x, x.y))
                        exc = np.hypot(x.x, x.y) - v.fence_at(vi, ang)
                        np.add.at(cells[i], _cell_index(ang.to_numpy(), exc.to_numpy()), x.exp.to_numpy())
            if n % 10 == 0:
                log(f"  v0 day {n + 1}/{len(days)}")
    finally:
        store.load = orig
    return pd.DataFrame(out, columns=["lam_hr", "lam_foul"]), cells
