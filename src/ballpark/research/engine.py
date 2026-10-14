"""Backtest engine: builds every window statistic once, then evaluates any hyper-parameter
setting of the candidate models as cheap array maths. Strictly walk-forward: every statistic
for a game on day D only uses pitches from [D-365, D-1], and the HR classifier is trained
only on seasons <= `train_through`."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from .dataset import Venues, build, hr_truth
from .hrmodel import HRClassifier, venue_matrix
from .windows import WindowSum

WINDOW = 365
ANG_EDGES = np.arange(-45, 46, 5.0)          # 18 angle bins, outer bins open-ended
DEPTH_EDGES = np.array([0, 10, 20, 30, 50, np.inf])  # ft beyond the wall
NA, ND = len(ANG_EDGES) - 1, len(DEPTH_EDGES) - 1
NCELL = NA * ND


@dataclass
class Params:
    k_bat: float = 150.0      # batter HR pseudo-PA toward league (park-adjusted)
    k_pit: float = 250.0      # pitcher HR pseudo-PA
    k_bat_foul: float = 100.0
    k_pit_foul: float = 250.0
    sp_default: float = 0.58  # starter share of PA when too few starts
    ang_sd: float = 3.0
    depth_sd: float = 8.0
    out_scale: float = 10.0   # depth-beyond-wall scale for balls that need to "carry"
    phys_depth: bool = True   # place balls using carry predicted from EV/LA instead of the observed distance
    spatial_prior_hr: float = 20.0
    use_pitcher: bool = True
    use_starter_share: bool = True
    hr_blend: float = 0.0     # weight on the batter's actual HR/PA (park-adjusted) vs xHR
    recal: bool = True        # scale by the league's recent actual/expected HR ratio (environment drift)
    recal_prior: float = 40.0  # pseudo-HR toward ratio 1 (xHR models) / 365d rate (rate models)
    park_resid: bool = False  # walk-forward per-park actual/expected HR correction (wind, climate, wall height)
    park_prior_hr: float = 100.0
    platoon: bool = True      # batter-vs-pitcher-hand multiplier relative to the batter's own PA mix


def _cell_index(angle, excess):
    a = np.clip(np.floor((np.asarray(angle) + 45) / 5), 0, NA - 1).astype(int)
    d = np.clip(np.searchsorted(DEPTH_EDGES, np.maximum(excess, 0), "right") - 1, 0, ND - 1)
    return a * ND + d


class Engine:
    def __init__(self, test_start, test_end, train_through, with_dist=True, use_venue=True, verbose=True):
        self.log = print if verbose else (lambda *a, **k: None)
        self.venues = Venues()
        self.test_start, self.test_end, self.train_through = test_start, test_end, train_through
        self.log("building tables ...")
        t = build("2021-04-01", test_end, self.venues)
        self.pa, self.balls, self.lineups = t["pa"], t["balls"], t["lineups"]
        self.nv = len(self.venues.ids)
        self.log(f"  {len(self.pa):,} PA, {len(self.balls):,} batted balls, {len(self.lineups):,} lineup slots")

        tr = self.balls[self.balls.game_year <= train_through]
        self.log(f"training HR classifier on {len(tr):,} balls (<= {train_through}) ...")
        self.clf = HRClassifier(self.venues, with_dist=with_dist, use_venue=use_venue).fit(tr)
        self._fit_carry(tr)
        self.M = venue_matrix(self.clf, self.balls)           # [n_balls, n_venues]
        self.Mn = self.M.mean(axis=1)                         # park-neutral P(HR)
        self._priors()
        self._windows()

    def _fit_carry(self, tr):
        """Expected carry (ft) from exit velocity + launch angle: where a ball *would* land if it
        cleared the wall, independent of where this particular ball was caught."""
        def X(d):
            return np.column_stack([d.launch_speed, d.launch_angle])
        fly = tr[tr.launch_angle.between(15, 55) & (tr.launch_speed >= 75)]
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=0).fit(X(fly), fly.dist)
        self.dhat = m.predict(X(self.balls))
        hrs = tr[tr.hr == 1]
        self.carry_resid_sd = float(np.std(hrs.dist - m.predict(X(hrs))))

    # ---------------- window statistics ----------------
    def _windows(self):
        pa, b = self.pa, self.balls
        pav = np.column_stack([np.ones(len(pa)), pa.hr, pa.fouls])
        self.w_bat = WindowSum(pa.batter, pa.dn, pav)
        self.w_pit = WindowSum(pa.pitcher, pa.dn, pav)
        self.w_lg = WindowSum(np.zeros(len(pa), int), pa.dn, pav)
        self.w_bat_f = WindowSum(b.batter, b.dn, self.M)
        self.w_pit_f = WindowSum(b.pitcher, b.dn, self.Mn)
        self.w_lg_f = WindowSum(np.zeros(len(b), int), b.dn, self.M)
        self.w_lg_fn = WindowSum(np.zeros(len(b), int), b.dn, self.Mn)
        # league actual-vs-expected HR, for the walk-forward environment recalibration
        act = self.M[np.arange(len(b)), b.vi.to_numpy()].astype(float)
        self.w_lg_act = WindowSum(np.zeros(len(b), int), b.dn, np.column_stack([act, b.hr]))
        self.w_park_act = WindowSum(b.vi, b.dn, np.column_stack([act, b.hr]))
        # platoon: expected league HR multiplier for each trailing PA of the batter
        mult = np.array([self.m_plat.get((st, th), 1.0) for st, th in zip(pa.stand, pa.p_throws)])
        self.w_bat_plat = WindowSum(pa.batter, pa.dn, mult)
        # park-level HR/PA history for the simple park-factor baseline
        self.w_park = WindowSum(pa.vi, pa.dn, pav)
        # starter workload: batters faced per start
        st = (self.lineups[["game_pk", "sp", "dn"]].drop_duplicates()
              .merge(pa.groupby(["game_pk", "pitcher"]).size().rename("bf").reset_index()
                     .rename(columns={"pitcher": "sp"}), on=["game_pk", "sp"], how="left"))
        st["bf"] = st.bf.fillna(0)
        self.w_start = WindowSum(st.sp.astype("int64"), st.dn, np.column_stack([st.bf, np.ones(len(st))]))

    def _priors(self):
        L, pa = self.lineups, self.pa
        tr = L[(L.year <= self.train_through) & (L.year >= 2022)]
        self.pa_slot = tr.groupby(["slot", "home"]).pa.mean()
        self.team_pa = tr.groupby(["game_pk", "side"]).pa.sum().mean()
        # park spatial prior from training-period HRs
        h = hr_truth(self.balls[self.balls.game_year <= self.train_through], self.venues)
        self.hr_cell = _cell_index(h.angle.to_numpy(), (h.dist - h.fence).to_numpy())
        lg = np.bincount(self.hr_cell, minlength=NCELL) + 1.0
        self.P_league = lg / lg.sum()
        self._park_counts = {}
        for v, g in h.groupby("vi"):
            c = np.bincount(_cell_index(g.angle.to_numpy(), (g.dist - g.fence).to_numpy()), minlength=NCELL)
            self._park_counts[v] = c
        # platoon table: HR/PA by (batter stand, pitcher hand), relative to all PA
        trp0 = pa[(pa.year <= self.train_through)]
        base = trp0.hr.mean()
        self.m_plat = (trp0.groupby(["stand", "p_throws"]).hr.mean() / base).to_dict()
        mix = trp0.groupby(["stand", "p_throws"]).size()
        self.m_mix = {st: sum(self.m_plat[(st, h)] * mix[(st, h)] for h in ("L", "R") if (st, h) in mix.index)
                      / sum(mix[(st, h)] for h in ("L", "R") if (st, h) in mix.index) for st in ("L", "R")}
        # park HR factor for the baseline model (shrunk HR/PA ratio over training years)
        trp = pa[pa.year <= self.train_through]
        lg_rate = trp.hr.mean()
        g = trp.groupby("vi").hr.agg(["sum", "size"])
        self.park_factor = np.ones(self.nv)
        for v, r in g.iterrows():
            self.park_factor[int(v)] = (r["sum"] + 3000 * lg_rate) / ((r["size"] + 3000) * lg_rate)

    def park_prior(self, vi, alpha):
        c = self._park_counts.get(vi, np.zeros(NCELL))
        p = c + alpha * self.P_league
        return p / p.sum()

    # ---------------- per-lineup-slot features ----------------
    def test_rows(self) -> pd.DataFrame:
        L = self.lineups
        m = (L.dn >= (pd.Timestamp(self.test_start) - pd.Timestamp("2000-01-01")).days) & \
            (L.dn <= (pd.Timestamp(self.test_end) - pd.Timestamp("2000-01-01")).days)
        return L[m & L.sp.notna() & L.pa.notna()].reset_index(drop=True)

    def stats(self, R: pd.DataFrame) -> dict:
        d1, d0 = R.dn.to_numpy(), R.dn.to_numpy() - WINDOW
        b, p, vi = R.batter.to_numpy(), R.sp.to_numpy().astype("int64"), R.vi.to_numpy().astype(int)
        z = np.zeros(len(R), int)
        s = {}
        s["bat"] = self.w_bat.query(b, d0, d1)            # PA, HR, FOUL
        s["pit"] = self.w_pit.query(p, d0, d1)
        s["lg"] = self.w_lg.query(z, d0, d1)
        s["bat_f"] = self.w_bat_f.query(b, d0, d1)[np.arange(len(R)), vi]       # Σ P(HR) in game park
        s["lg_f"] = self.w_lg_f.query(z, d0, d1)[np.arange(len(R)), vi]
        s["pit_fn"] = self.w_pit_f.query(p, d0, d1)[:, 0]
        s["lg_fn"] = self.w_lg_fn.query(z, d0, d1)[:, 0]
        s["start"] = self.w_start.query(p, d0, d1)                                 # BF, starts
        s["lg120"] = self.w_lg.query(z, d1 - self.RECAL, d1)
        s["lg_act120"] = self.w_lg_act.query(z, d1 - self.RECAL, d1)
        s["park730"] = self.w_park_act.query(vi, d1 - 730, d1)
        s["lg730"] = self.w_lg_act.query(z, d1 - 730, d1)
        s["plat_trail"] = self.w_bat_plat.query(b, d0, d1)[:, 0]
        s["pa_exp"] = np.array([self.pa_slot.get((sl, h), 4.0) for sl, h in zip(R.slot, R.home)])
        s["park_factor"] = self.park_factor[vi]
        # this game's platoon multiplier: starter's hand for his share of PA, league mix otherwise
        st = R.stand.to_numpy(); hand = R.sp_hand.to_numpy()
        s["m_sp"] = np.array([self.m_plat.get((a, h), 1.0) for a, h in zip(st, hand)])
        s["m_mix"] = np.array([self.m_mix.get(a, 1.0) for a in st])
        return s

    RECAL = 120

    def recal_factor(self, s, P: Params, xhr: bool):
        if not P.recal:
            return 1.0
        if xhr:   # league actual HR / model-expected HR over the recent window
            return (s["lg_act120"][:, 1] + P.recal_prior) / (s["lg_act120"][:, 0] + P.recal_prior)
        # recent league HR count vs what the 365d rate implies; the pseudo-HR prior keeps the
        # factor sane in the first days of a season when the window is nearly empty
        r365 = s["lg"][:, 1] / s["lg"][:, 0]
        expected = s["lg120"][:, 0] * r365
        return (s["lg120"][:, 1] + P.recal_prior) / (expected + P.recal_prior)

    def park_resid_factor(self, s, P: Params):
        if not P.park_resid:
            return 1.0
        pk, lg = s["park730"], s["lg730"]
        c = P.park_prior_hr
        return ((pk[:, 1] + c) / (pk[:, 0] + c)) / (lg[:, 1] / lg[:, 0])

    def platoon_factor(self, s, P: Params, share):
        if not P.platoon:
            return 1.0
        pa_b = np.maximum(s["bat"][:, 0], 1)
        trail = s["plat_trail"] / pa_b
        trail = np.where(s["bat"][:, 0] >= 30, trail, 1.0)
        game = share * s["m_sp"] + (1 - share) * s["m_mix"]
        return game / np.where(s["bat"][:, 0] >= 30, trail, game)

    def sp_share(self, s, P: Params):
        if not P.use_starter_share:
            return np.ones(len(s["pa_exp"]))
        bf, n = s["start"][:, 0], s["start"][:, 1]
        avg = np.where(n >= 3, bf / np.maximum(n, 1), P.sp_default * self.team_pa)
        return np.clip(avg / self.team_pa, 0.25, 0.85)

    # ---------------- HR / foul rate models ----------------
    def lam_hr(self, R, s, P: Params, model="v1"):
        pa_b, hr_b = s["bat"][:, 0], s["bat"][:, 1]
        lg_rate = s["lg"][:, 1] / s["lg"][:, 0]
        if model == "league":
            return s["pa_exp"] * lg_rate * self.recal_factor(s, P, False)
        if model == "batter":
            rate = (hr_b + P.k_bat * lg_rate) / (pa_b + P.k_bat)
            return s["pa_exp"] * rate * self.recal_factor(s, P, False)
        if model == "rate":     # simple: batter HR/PA x park factor x pitcher HR/PA ratio
            rate = (hr_b + P.k_bat * lg_rate) / (pa_b + P.k_bat) * s["park_factor"] * self.recal_factor(s, P, False)
            pf = ((s["pit"][:, 1] + P.k_pit * lg_rate) / (s["pit"][:, 0] + P.k_pit)) / lg_rate
            share = self.sp_share(s, P)
            return s["pa_exp"] * rate * ((share * pf + (1 - share)) if P.use_pitcher else 1)
        # v1: park-specific expected HR from the batter's own batted balls
        mu = s["lg_f"] / s["lg"][:, 0]                        # league xHR/PA in this park
        x_rate = (s["bat_f"] + P.k_bat * mu) / (pa_b + P.k_bat)
        if P.hr_blend > 0:  # blend in actual HR/PA, park-adjusted by the (neutral->park) xHR ratio
            act = (hr_b + P.k_bat * lg_rate) / (pa_b + P.k_bat) * (mu / (s["lg_fn"] / s["lg"][:, 0]))
            x_rate = (1 - P.hr_blend) * x_rate + P.hr_blend * act
        mun = s["lg_fn"] / s["lg"][:, 0]
        pf = ((s["pit_fn"] + P.k_pit * mun) / (s["pit"][:, 0] + P.k_pit)) / mun
        share = self.sp_share(s, P)
        pit = (share * pf + (1 - share)) if P.use_pitcher else 1.0
        return s["pa_exp"] * x_rate * pit * self.recal_factor(s, P, True) * self.platoon_factor(s, P, share) * self.park_resid_factor(s, P)

    def lam_foul(self, R, s, P: Params, model="v1"):
        lg_rate = s["lg"][:, 2] / s["lg"][:, 0]
        pa_b, f_b = s["bat"][:, 0], s["bat"][:, 2]
        if model == "league":
            return s["pa_exp"] * lg_rate
        rate = (f_b + P.k_bat_foul * lg_rate) / (pa_b + P.k_bat_foul)
        if model == "batter":
            return s["pa_exp"] * rate
        pf = ((s["pit"][:, 2] + P.k_pit_foul * lg_rate) / (s["pit"][:, 0] + P.k_pit_foul)) / lg_rate
        share = self.sp_share(s, P)
        return s["pa_exp"] * rate * ((share * pf + (1 - share)) if P.use_pitcher else 1.0)

    # ---------------- spatial HR model ----------------
    def _ball_cells(self, f, ang, dist, fence, P: Params):
        """Cell distribution of each ball *given it leaves the park*: [k, NCELL]."""
        a = (norm.cdf((ANG_EDGES[None, 1:] - ang[:, None]) / P.ang_sd)
             - norm.cdf((ANG_EDGES[None, :-1] - ang[:, None]) / P.ang_sd))
        a[:, 0] += norm.cdf((ANG_EDGES[0] - ang) / P.ang_sd)
        a[:, -1] += 1 - norm.cdf((ANG_EDGES[-1] - ang) / P.ang_sd)
        e = dist - fence
        lo, hi = DEPTH_EDGES[None, :-1], DEPTH_EDGES[None, 1:]
        cdf = lambda x: np.where(np.isinf(x), 1.0, norm.cdf((x - e[:, None]) / P.depth_sd))
        tn = np.clip(cdf(hi) - cdf(lo), 0, None)
        tn_sum = tn.sum(1, keepdims=True)
        tn = tn / np.where(tn_sum > 0, tn_sum, 1)
        ex = np.exp(-lo / P.out_scale) - np.where(np.isinf(hi), 0, np.exp(-hi / P.out_scale))
        d = np.where((e >= 0)[:, None], tn, ex)
        return (a[:, :, None] * d[:, None, :]).reshape(len(f), NCELL)

    def cells_v1(self, R, s, lam, P: Params, with_batter=True):
        """Expected HR per cell for each lineup slot: [n_rows, NCELL]; sums to `lam` per row."""
        n = len(R)
        out = np.zeros((n, NCELL))
        d1, d0 = R.dn.to_numpy(), R.dn.to_numpy() - WINDOW
        lo, hi = self.w_bat_f.span(R.batter.to_numpy(), d0, d1)
        vi = R.vi.to_numpy().astype(int)
        pa_b = s["bat"][:, 0]
        mu = s["lg_f"] / s["lg"][:, 0]
        ang_all = self.balls.angle.to_numpy()
        dist_all = self.dhat if P.phys_depth else self.balls.dist.to_numpy()
        for i in range(n):
            v = vi[i]
            prior = P.k_bat * mu[i] * self.park_prior(v, P.spatial_prior_hr)
            S = np.zeros(NCELL)
            if with_batter and hi[i] > lo[i]:
                idx = self.w_bat_f.order[lo[i]:hi[i]]
                f = self.M[idx, v]
                keep = f > 1e-3
                if keep.any():
                    idx, f = idx[keep], f[keep].astype(float)
                    fence = self.venues.fence_at(v, ang_all[idx])
                    S = (f[:, None] * self._ball_cells(f, ang_all[idx], dist_all[idx], fence, P)).sum(0)
            tot = S.sum() + prior.sum()
            out[i] = (S + prior) / (pa_b[i] + P.k_bat) if tot > 0 else 0
        row_sum = out.sum(1, keepdims=True)
        return out / np.where(row_sum > 0, row_sum, 1) * lam[:, None]

    def cells_prior_only(self, R, lam, P: Params):
        c = np.vstack([self.park_prior(int(v), P.spatial_prior_hr) for v in R.vi])
        return c * lam[:, None]

    def actual_cells(self, R):
        """Observed HRs by lineup members -> list of (row index of game-side batter, cell)."""
        h = self.balls[(self.balls.hr == 1)]
        key = R.reset_index().set_index(["game_pk", "batter"])["index"]
        h = h.join(key.rename("row"), on=["game_pk", "batter"], how="inner")
        cell = _cell_index(h.angle.to_numpy(), (h.dist - h.fence).to_numpy())
        return pd.DataFrame({"row": h.row.to_numpy(), "game_pk": h.game_pk.to_numpy(), "cell": cell})
