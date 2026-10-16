"""One-off ablations of the ball-level HR classifier (run locally, see docs/MODELING.md)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from .dataset import Venues, build
from .games import load_games
from .hrmodel import CANDIDATE, features


def run(test_year=2024, log=print):
    v = Venues()
    t = build("2021-04-01", f"{test_year}-10-31", v)
    b = t["balls"]
    g = load_games().set_index("game_pk")
    b["temp"] = b.game_pk.map(g.temp)
    b = b[CANDIDATE(b)].reset_index(drop=True)
    tr, te = b[b.game_year < test_year], b[b.game_year == test_year]
    rows = []
    for name, with_dist, use_venue, temp in (("base (no distance)", False, True, False),
                                             ("+ hit distance", True, True, False),
                                             ("no venue id (fence/elev only)", False, False, False),
                                             ("+ game temperature", False, True, True)):
        def X(d):
            x = features(d, d.vi.to_numpy(), v, with_dist)
            if use_venue:
                x["venue"] = pd.Categorical(x.venue, categories=range(len(v.ids)))
            else:
                x = x.drop(columns="venue")
            if temp:
                x["temp"] = d.temp.to_numpy()
            return x
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                           l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
        m.fit(X(tr), tr.hr)
        p = m.predict_proba(X(te))[:, 1]
        rows.append({"classifier": name, "logloss": log_loss(te.hr, p), "auc": roc_auc_score(te.hr, p),
                     "pred_hr": p.sum(), "actual_hr": int(te.hr.sum())})
        log(rows[-1])
    return pd.DataFrame(rows)
