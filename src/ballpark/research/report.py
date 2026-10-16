"""Render output/backtests/<name>/report.md + PNGs from a finished backtest run."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import OUTPUT


def _md(df, nd=4):
    return df.round(nd).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(nd).to_string(index=False)


def render(name):
    d = OUTPUT / "backtests" / name
    meta = json.loads((d / "meta.json").read_text())
    t = {p.stem: pd.read_csv(p) for p in d.glob("*.csv")}
    # calibration plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, lab, c in (("calibration_batter", "batter-only baseline", "#999"), ("calibration_v1", "v1", "#d1495b")):
        ax[0].plot(t[k].pred, t[k].actual, "o-", label=lab, color=c)
    m = max(t["calibration_v1"].pred.max(), t["calibration_v1"].actual.max()) * 1.05
    ax[0].plot([0, m], [0, m], "k:", lw=1); ax[0].set_xlabel("predicted P(HR >= 1)"); ax[0].set_ylabel("observed")
    ax[0].set_title("Batter-game HR calibration (deciles)"); ax[0].legend()
    c = t["cells_total"]
    for a, col, lab in ((ax[1], "angle_bin", "spray-angle bin (5 deg, LF -> RF)"), (ax[2], "depth_bin", "ft beyond wall (0-10,10-20,20-30,30-50,50+)")):
        g = c.groupby(col)[["actual", "v1", "park_prior"]].sum()
        a.bar(g.index - 0.2, g.actual, 0.4, label="actual HR", color="#555")
        a.bar(g.index + 0.2, g.v1, 0.4, label="v1 expected", color="#d1495b")
        a.plot(g.index, g.park_prior, "k.", label="park prior")
        a.set_xlabel(lab); a.set_title("HR location: expected vs actual"); a.legend()
    fig.tight_layout(); fig.savefig(d / "calibration.png", dpi=110); plt.close(fig)
    parts = [f"# Backtest `{name}`\n",
             f"Test window {meta['test'][0]} .. {meta['test'][1]} ({meta['games']:,} games, {meta['rows']:,} lineup slots). "
             f"HR classifier trained on seasons <= {meta['train_through']}. v0 evaluated on every {meta['every']}rd/th day.\n",
             f"Classifier on test-window balls: {json.dumps({k: round(v, 4) for k, v in meta['classifier_test'].items()})}\n"]
    for k, title in (("hr_full", "HR count per batter-game (all games)"), ("hr_sample", "HR count - same sample v0 was run on"),
                     ("bootstrap", "Paired bootstrap: mean Poisson-NLL difference, v1 minus other (negative = v1 better)"),
                     ("foul", "Foul count per batter-game"), ("spatial", "HR location (nats per actual HR, lower is better)"),
                     ("spatial_sample", "HR location - v0 sample"), ("by_park", "By park")):
        if k in t:
            parts.append(f"## {title}\n\n{_md(t[k])}\n")
    parts.append("![calibration](calibration.png)\n")
    (d / "report.md").write_text("\n".join(parts))
    return d / "report.md"
