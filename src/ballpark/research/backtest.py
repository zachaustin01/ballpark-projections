"""Walk-forward backtest + report. Run locally only:  uv run ballpark research backtest ..."""
import itertools
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from ..paths import OUTPUT
from . import metrics as M
from .engine import NCELL, Engine, Params, _cell_index

HR_MODELS = ["league", "batter", "rate", "v1"]


def _count_rows(y, lam, dn, label, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    L = M.count_table(y[mask], lam[mask], dn[mask])
    return {"model": label, "n": int(mask.sum()), **M.summary(y[mask], lam[mask], L)}, L


def fit_params(eng, R, s, base: Params, log=print) -> Params:
    """Coarse coordinate-wise search on the *validation* window only (Poisson NLL)."""
    y = R.hr.to_numpy()
    P = base
    grid = {"k_bat": [50, 100, 150, 250, 400], "k_pit": [100, 250, 500, 1000]}
    for _ in range(2):
        for name, vals in grid.items():
            best = min(vals, key=lambda v: M.pois_nll(y, eng.lam_hr(R, s, replace(P, **{name: v}), "v1")).mean())
            P = replace(P, **{name: best})
    yf = R.fouls.to_numpy()
    for name, vals in {"k_bat_foul": [30, 60, 100, 200, 400], "k_pit_foul": [100, 250, 500, 1000]}.items():
        best = min(vals, key=lambda v: M.pois_nll(yf, eng.lam_foul(R, s, replace(P, **{name: v}))).mean())
        P = replace(P, **{name: best})
    log(f"tuned: {asdict(P)}")
    return P


def tune_spatial(eng, R, s, P, log=print):
    """Pick the depth/angle kernel + prior strength that minimise game-level location
    log-loss, on every 5th game of the validation window."""
    m = (R.game_pk.rank(method="dense").to_numpy() % 5 == 0)
    Rm = R[m].reset_index(drop=True)
    sm = eng.stats(Rm)
    lam = eng.lam_hr(Rm, sm, P, "v1")
    a = eng.actual_cells(Rm)
    gi = pd.factorize(Rm.game_pk)[0]
    best = None
    log(f"carry-model residual sd on training HRs: {eng.carry_resid_sd:.1f} ft")
    for depth_sd, sd, prior, phys in itertools.product([8, 12, 16, 22], [3.0, 5.0], [5, 20], [True, False]):
        Q = replace(P, depth_sd=depth_sd, ang_sd=sd, spatial_prior_hr=prior, phys_depth=phys)
        g, b = M.spatial_scores(eng.cells_v1(Rm, sm, lam, Q), gi, a, NCELL)
        log(f"  spatial phys={phys} depth_sd={depth_sd} ang_sd={sd} prior={prior}: game {g.mean():.4f}  batter {b.mean():.4f}")
        if best is None or g.mean() < best[0]:
            best = (g.mean(), Q)
    return best[1]


def run(name, test_start, test_end, train_through, every=7, params: Params | None = None,
        tune=False, with_v0=True, with_dist=False, log=print):
    out = OUTPUT / "backtests" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eng = Engine(test_start, test_end, train_through, with_dist=with_dist, verbose=True)
    R = eng.test_rows()
    s = eng.stats(R)
    log(f"{len(R):,} lineup slots, {R.game_pk.nunique():,} games in {test_start}..{test_end}")
    P = params or Params()
    if tune:
        P = fit_params(eng, R, s, P, log)
        P = tune_spatial(eng, R, s, P, log)
        (out / "params.json").write_text(json.dumps(asdict(P), indent=1))
    dn = R.dn.to_numpy()
    y, yf = R.hr.to_numpy(), R.fouls.to_numpy()

    lam = {m: eng.lam_hr(R, s, P, m) for m in HR_MODELS}
    lam["v1 (no pitcher)"] = eng.lam_hr(R, s, replace(P, use_pitcher=False), "v1")
    lam["v1 (pitcher, no starter share)"] = eng.lam_hr(R, s, replace(P, use_starter_share=False), "v1")
    lam["v1 (+25% actual HR blend)"] = eng.lam_hr(R, s, replace(P, hr_blend=0.25), "v1")
    flam = {m: eng.lam_foul(R, s, P, m) for m in ("league", "batter", "v1")}

    # ---- classifier diagnostics (ball level, test window) ----
    tb = eng.balls[(eng.balls.dn >= R.dn.min()) & (eng.balls.dn <= R.dn.max())]
    clf_diag = eng.clf.score(tb)

    # ---- spatial ----
    games = pd.factorize(R.game_pk)[0]
    act = eng.actual_cells(R)
    cells = {"uniform": np.full((len(R), NCELL), 1.0 / NCELL) * lam["v1"][:, None],
             "all-park prior": np.tile(eng.P_league, (len(R), 1)) * lam["v1"][:, None],
             "park prior": eng.cells_prior_only(R, lam["v1"], P),
             "v1": eng.cells_v1(R, s, lam["v1"], P)}

    # ---- v0 on a 1-in-`every` day sample ----
    sub = (dn % every == 0)
    v0 = {}
    if with_v0:
        from . import legacy
        Rs = R[sub].reset_index(drop=True)
        for label, legacy_norm in (("v0 (original)", True), ("v0 (league-normalised)", False)):
            log(f"running {label} on {sub.sum():,} lineup slots ...")
            res, c = legacy.run(eng, Rs, legacy_norm, log)
            full_l = np.zeros(len(R)); full_f = np.zeros(len(R)); full_c = np.zeros((len(R), NCELL))
            full_l[sub], full_f[sub], full_c[sub] = res.lam_hr.to_numpy(), res.lam_foul.to_numpy(), c
            v0[label] = (full_l, full_f, full_c)
            lam[label], flam[label], cells[label] = full_l, full_f, full_c

    # ---- tables ----
    tabs = {}
    rows, losses = [], {}
    for m, l in lam.items():
        if m.startswith("v0"):
            continue
        r, L = _count_rows(y, l, dn, m); rows.append(r); losses[m] = L
    tabs["hr_full"] = pd.DataFrame(rows)
    rows, losses_sub = [], {}
    for m, l in lam.items():
        r, L = _count_rows(y[sub], l[sub], dn[sub], m); rows.append(r); losses_sub[m] = L
    tabs["hr_sample"] = pd.DataFrame(rows)
    rows = []
    for m, l in flam.items():
        if m.startswith("v0"):
            r, _ = _count_rows(yf[sub], l[sub], dn[sub], m); rows.append(r)
        else:
            r, _ = _count_rows(yf, l, dn, m); rows.append(r)
    tabs["foul"] = pd.DataFrame(rows)
    rows = []
    for m, l in flam.items():
        r, _ = _count_rows(yf[sub], l[sub], dn[sub], m); rows.append(r)
    tabs["foul_sample"] = pd.DataFrame(rows)

    rows = []
    gs, bs = {}, {}
    for m, c in cells.items():
        use = np.ones(len(R), bool) if not m.startswith("v0") else sub
        a = act[use[act.row.to_numpy()]]
        g, b = M.spatial_scores(c, games, a, NCELL)
        gs[m], bs[m] = g, b
        rows.append({"model": m, "n_HR": len(a), "game_logloss": g.mean(), "batter_logloss": b.mean()})
    tabs["spatial"] = pd.DataFrame(rows)
    # same-sample comparison for the models vs v0
    rows = []
    a = act[sub[act.row.to_numpy()]]
    for m, c in cells.items():
        g, b = M.spatial_scores(c, games, a, NCELL)
        rows.append({"model": m, "n_HR": len(a), "game_logloss": g.mean(), "batter_logloss": b.mean()})
    tabs["spatial_sample"] = pd.DataFrame(rows)

    ct = pd.DataFrame({"cell": np.arange(NCELL), "angle_bin": np.arange(NCELL) // 5, "depth_bin": np.arange(NCELL) % 5,
                       "actual": np.bincount(act.cell, minlength=NCELL),
                       "v1": cells["v1"].sum(0), "park_prior": cells["park prior"].sum(0)})
    ct.to_csv(out / "cells_total.csv", index=False)

    # paired bootstrap v1 vs others on the HR nll
    boots = []
    for m in lam:
        if m == "v1":
            continue
        if m.startswith("v0"):
            d, lo, hi = M.paired_bootstrap(losses_sub["v1"]["nll"], losses_sub[m]["nll"], dn[sub])
            n = int(sub.sum())
        else:
            d, lo, hi = M.paired_bootstrap(losses["v1"]["nll"], losses[m]["nll"], dn)
            n = len(R)
        boots.append({"vs": m, "n": n, "mean_nll_diff": d, "ci_lo": lo, "ci_hi": hi})
    tabs["bootstrap"] = pd.DataFrame(boots)

    # calibration
    cal = M.calibration(losses["v1"]["p"], losses["v1"]["y1"])
    cal0 = M.calibration(losses["batter"]["p"], losses["batter"]["y1"])
    tabs["calibration_v1"], tabs["calibration_batter"] = cal, cal0
    # by park
    byp = (pd.DataFrame({"venue": R.vi.map(dict(enumerate(eng.venues.df.name))), "y": y,
                         "v1": lam["v1"], "batter": lam["batter"]}).groupby("venue")
           .agg(actual=("y", "sum"), v1=("v1", "sum"), batter=("batter", "sum"), n=("y", "size")))
    byp = byp[byp.n > 300].assign(v1_err=lambda d: d.v1 - d.actual, batter_err=lambda d: d.batter - d.actual)
    tabs["by_park"] = byp.reset_index()
    for k, t in tabs.items():
        t.to_csv(out / f"{k}.csv", index=False)
    pd.DataFrame({"game_pk": R.game_pk, "batter": R.batter, "side": R.side, "slot": R.slot, "hr": y, "fouls": yf,
                  **{f"hr_{k}": v for k, v in lam.items()}, **{f"foul_{k}": v for k, v in flam.items()}}
                 ).to_parquet(out / "predictions.parquet", index=False)
    meta = {"name": name, "test": [test_start, test_end], "train_through": train_through, "every": every,
            "params": asdict(P), "classifier_test": clf_diag, "rows": len(R), "games": int(R.game_pk.nunique()),
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    log(json.dumps(meta, indent=1, default=float))
    for k, t in tabs.items():
        log(f"\n== {k}\n{t.round(4).to_string(index=False)}")
    return eng, R, s, P, tabs
