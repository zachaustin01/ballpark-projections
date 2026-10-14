"""Streamlit UI:  uv run streamlit run src/ballpark/app.py"""
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ballpark import statcast as sc
from ballpark.ballparks import load_ballparks
from ballpark.geometry import ANGLES, fence_distance, to_xy
from ballpark.model import project_game
from ballpark.schedule import get_slate

st.set_page_config(page_title="Ballpark projections", layout="wide")


@st.cache_data(ttl=600, show_spinner="Loading schedule...")
def slate(d: str):
    return get_slate(d)


@st.cache_data(show_spinner="Pulling Statcast + projecting (first run per game is slow)...")
def project(d: str, game_pk: int, days: int):
    g = next(g for g in get_slate(d) if g["game_pk"] == game_pk)
    park = load_ballparks().set_index("venue_id").loc[g["venue_id"]]
    return project_game(g, park, d, days)


@st.cache_data(show_spinner="Summarising players...")
def player_stats(d: str, days: int, ids: tuple, kind: str):
    start, end = sc.window(d, days)
    pull = sc.pull_pitcher if kind == "pitcher" else sc.pull_batter
    return {i: sc.summarize(pull(i, start, end), kind) for i in ids}


def draw_field(ax, park, foul=False):
    a = np.linspace(-45, 45, 91)
    wx, wy = to_xy(a, fence_distance(a, park))
    ax.plot(wx, wy, color="#2e7d32", lw=2)
    for sgn in (-1, 1):
        ex, ey = to_xy(sgn * 45, fence_distance(sgn * 45, park))
        ax.plot([0, ex], [0, ey], color="white", lw=1, alpha=.8)
    ax.fill([0, 63.6, 0, -63.6], [0, 63.6, 127.3, 63.6], color="#c8a165", alpha=.5)
    ax.set_facecolor("#3a7d44" if not foul else "#2f6b39")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if foul:
        ax.set_xlim(-420, 420); ax.set_ylim(-120, 440)
    else:
        ax.set_xlim(-300, 300); ax.set_ylim(-20, 470)


def heat(res, kind, park, bins):
    d = res[res.kind == kind]
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_field(ax, park, foul=(kind == "FOUL"))
    if len(d):
        hb = ax.hexbin(d.x, d.y, C=d.exp, reduce_C_function=np.sum, gridsize=bins,
                       mincnt=1, cmap="YlOrRd", alpha=.85,
                       extent=(-420, 420, -120, 440) if kind == "FOUL" else (-300, 300, -20, 470))
        fig.colorbar(hb, ax=ax, shrink=.7, label="expected count per cell")
    return fig


st.title("Ballpark HR / foul-ball projections")
c1, c2, c3 = st.columns([1, 2, 1])
day = c1.date_input("Date", date.today())
d = day.isoformat()
days = c3.slider("Statcast lookback (days)", 30, 730, 365, 30)
games = slate(d)
if not games:
    st.info("No games on this date.")
    st.stop()


def label(g):
    sources = {g[x]["lineup_source"] for x in ("away", "home")}
    ready = ("no lineups" if not (g["away"]["lineup"] and g["home"]["lineup"])
             else "projected lineups" if sources - {"posted"} else "lineups posted")
    return f"{g['away']['team']} @ {g['home']['team']}  ({ready})"


g = games[c2.selectbox("Game", range(len(games)), format_func=lambda i: label(games[i]))]
park = load_ballparks().set_index("venue_id").loc[g["venue_id"]]
st.subheader(f"{g['away']['team']} @ {g['home']['team']} — {park['name']}")
st.caption(f"{g['status']} · capacity {park['capacity']:,} · LF {park['leftLine']} / CF {park['center']} / "
           f"RF {park['rightLine']} ft · elevation {park['elevation_ft']} ft")

cols = st.columns(2)
for col, s in zip(cols, ("away", "home")):
    side = g[s]
    with col:
        st.markdown(f"**{side['team']}**")
        p = side["pitcher"]
        rows = []
        if p:
            st_ = player_stats(d, days, (p["id"],), "pitcher")[p["id"]]
            rows.append({"#": "SP", "name": p["name"], **st_})
        if side["lineup"]:
            bs = player_stats(d, days, tuple(b["id"] for b in side["lineup"]), "batter")
            rows += [{"#": i, "name": b["name"], **bs[b["id"]]} for i, b in enumerate(side["lineup"], 1)]
        else:
            st.caption("Lineup not posted and no prior lineup found.")
        if side["lineup"] and side["lineup_source"] != "posted":
            st.caption(f"Lineup {side['lineup_source']}")
        if rows:
            st.dataframe(pd.DataFrame(rows).astype({"#": str}), hide_index=True, use_container_width=True)

if not (g["away"]["lineup"] and g["home"]["lineup"]):
    st.warning("Projection needs a lineup for both teams.")
    st.stop()

res = project(d, g["game_pk"], days)
if res.empty:
    st.warning("No projection (insufficient Statcast data).")
    st.stop()

st.divider()
bins = st.slider("Heatmap resolution", 15, 60, 35)
t_hr, t_foul = st.tabs(["Home runs", "Foul balls (estimated)"])
for tab, kind in ((t_hr, "HR"), (t_foul, "FOUL")):
    with tab:
        sub = res[res.kind == kind]
        m1, m2 = st.columns([3, 2])
        m1.pyplot(heat(res, kind, park, bins))
        m2.metric(f"Expected {kind.lower()}s this game", f"{sub.exp.sum():.2f}")
        team_tot = sub.groupby("team").exp.sum().round(2).rename("expected")
        m2.dataframe(team_tot, use_container_width=True)
        z = sub.groupby("zone", as_index=False).exp.sum().sort_values("exp", ascending=False).head(12)
        m2.markdown("**Top zones**")
        m2.dataframe(z.assign(exp=z.exp.round(3)), hide_index=True, use_container_width=True)
        b = sub.groupby("batter", as_index=False).exp.sum().sort_values("exp", ascending=False).head(9)
        m2.markdown("**Top batters**")
        m2.dataframe(b.assign(exp=b.exp.round(3)), hide_index=True, use_container_width=True)
