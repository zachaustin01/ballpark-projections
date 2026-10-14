import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score

EPS_CELL = 1e-3


def pois_nll(y, lam):
    lam = np.clip(lam, 1e-9, None)
    return lam - y * np.log(lam) + gammaln(y + 1)


def pois_dev(y, lam):
    lam = np.clip(lam, 1e-9, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / lam), 0.0)
    return 2 * (t - (y - lam))


def bin_logloss(y1, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y1 * np.log(p) + (1 - y1) * np.log(1 - p))


def count_table(y, lam, dn) -> dict:
    """Per-row losses plus summary for a Poisson count model."""
    p = 1 - np.exp(-lam)
    y1 = (y > 0).astype(float)
    return {"nll": pois_nll(y, lam), "dev": pois_dev(y, lam), "ll1": bin_logloss(y1, p),
            "brier": (p - y1) ** 2, "p": p, "y1": y1}


def summary(y, lam, losses) -> dict:
    return {"pred_total": lam.sum(), "actual_total": float(y.sum()),
            "bias_%": 100 * (lam.sum() / max(y.sum(), 1) - 1),
            "poisson_nll": losses["nll"].mean(), "deviance": losses["dev"].mean(),
            "brier(P>=1)": losses["brier"].mean(), "logloss(P>=1)": losses["ll1"].mean(),
            "auc(P>=1)": roc_auc_score(losses["y1"], losses["p"]) if 0 < losses["y1"].sum() < len(y) else np.nan}


def calibration(p, y1, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"bin": q, "p": p, "y": y1}).groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return d.reset_index()


def paired_bootstrap(loss_a, loss_b, cluster, n=1000, seed=0):
    """Mean(loss_a - loss_b) with a 95% CI, resampling whole days (rows within a day are
    correlated). Negative = model A is better."""
    df = pd.DataFrame({"d": loss_a - loss_b, "c": cluster}).groupby("c").d.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), (n, len(df)))
    s, k = df["sum"].to_numpy(), df["size"].to_numpy()
    boot = s[idx].sum(1) / k[idx].sum(1)
    return float(s.sum() / k.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def spatial_scores(pred_cells, game_idx, act, ncell):
    """Game-level and batter-level HR-location log-loss (nats per actual HR).
    pred_cells: [rows, ncell] expected HR per cell; act: DataFrame[row, cell]."""
    uni = 1.0 / ncell
    ng = game_idx.max() + 1
    G = np.zeros((ng, ncell))
    np.add.at(G, game_idx, pred_cells)
    Gn = G / np.where(G.sum(1, keepdims=True) > 0, G.sum(1, keepdims=True), 1)
    Gn = np.where(G.sum(1, keepdims=True) > 0, Gn, uni)
    pg = (1 - EPS_CELL) * Gn[game_idx[act.row.to_numpy()], act.cell.to_numpy()] + EPS_CELL * uni
    rows = pred_cells[act.row.to_numpy()]
    tot = rows.sum(1)
    pb = np.where(tot > 0, rows[np.arange(len(act)), act.cell.to_numpy()] / np.where(tot > 0, tot, 1), uni)
    pb = (1 - EPS_CELL) * pb + EPS_CELL * uni
    return -np.log(pg), -np.log(pb)
