"""Per-ball P(home run | ball, park): gradient-boosted classifier on contact quality,
spray angle and park geometry. Evaluating one ball in every park gives the counterfactual
"would this have left in <park>?" that the batter-level rates are built from."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from .dataset import Venues

BASE = ["launch_speed", "launch_angle", "angle"]
CANDIDATE = (  # only balls that could plausibly leave the yard get a nonzero probability
    lambda d: (d.launch_angle.to_numpy() >= 10) & (d.launch_angle.to_numpy() <= 60)
    & (d.launch_speed.to_numpy() >= 70)
)


def features(balls: pd.DataFrame, vi, venues: Venues, with_dist=True) -> pd.DataFrame:
    """Features for `balls` as if hit in venue index(es) `vi` (scalar or per-row)."""
    vi = np.broadcast_to(np.asarray(vi), (len(balls),)).astype(int)
    ang = balls.angle.to_numpy()
    fence = venues.fence_at(vi, ang)
    f = pd.DataFrame({"ev": balls.launch_speed.to_numpy(), "la": balls.launch_angle.to_numpy(),
                      "ang": ang, "abs_ang": np.abs(ang), "fence": fence,
                      "elev": venues.elev[vi], "roof": venues.roof[vi], "venue": vi})
    if with_dist:
        f["dist"] = balls.dist.to_numpy()
        f["margin"] = f.dist - f.fence
    return f


class HRClassifier:
    def __init__(self, venues: Venues, with_dist=True, use_venue=True, max_iter=300, lr=0.06):
        self.venues, self.with_dist, self.use_venue = venues, with_dist, use_venue
        self.m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=31,
                                                min_samples_leaf=40, l2_regularization=1.0,
                                                categorical_features="from_dtype", random_state=0)

    def _x(self, balls, vi):
        x = features(balls, vi, self.venues, self.with_dist)
        if self.use_venue:
            x["venue"] = pd.Categorical(x.venue, categories=range(len(self.venues.ids)))
        else:
            x = x.drop(columns="venue")
        return x

    def fit(self, balls: pd.DataFrame):
        d = balls[CANDIDATE(balls)]
        self.m.fit(self._x(d, d.vi.to_numpy()), d.hr.to_numpy())
        return self

    def predict(self, balls, vi) -> np.ndarray:
        """P(HR) for each ball in venue(s) vi; zero for non-candidates."""
        out = np.zeros(len(balls))
        mask = CANDIDATE(balls)
        if mask.any():
            d = balls[mask]
            vv = np.broadcast_to(np.asarray(vi), (len(balls),))[mask]
            out[mask] = self.m.predict_proba(self._x(d, vv))[:, 1]
        return out

    def score(self, balls) -> dict:
        p = self.predict(balls, balls.vi.to_numpy())
        y = balls.hr.to_numpy()
        return {"logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "auc": roc_auc_score(y, p),
                "pred_hr": float(p.sum()), "actual_hr": int(y.sum())}


def venue_matrix(clf: HRClassifier, balls: pd.DataFrame, chunk=200_000) -> np.ndarray:
    """P(HR) of every ball in every venue: float32 [n_balls, n_venues]."""
    nv = len(clf.venues.ids)
    out = np.zeros((len(balls), nv), np.float32)
    cand = np.flatnonzero(CANDIDATE(balls))
    for v in range(nv):
        for s in range(0, len(cand), chunk):
            idx = cand[s:s + chunk]
            out[idx, v] = clf.m.predict_proba(clf._x(balls.iloc[idx], v))[:, 1]
    return out
