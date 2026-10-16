# Modeling & backtesting

Everything here is **local-only**. Nothing in `src/ballpark/research/` runs in CI (see "CI isolation").

## What is being predicted

| Target | Ground truth | Scored how |
|---|---|---|
| HR count / P(HR >= 1) per batter-game | `events == home_run` | Poisson NLL, Brier, log-loss, AUC, decile calibration |
| Where HRs land | `hc_x/hc_y` (angle) and `hit_distance_sc` (depth beyond the wall) of actual HRs | Log-loss (nats per HR) of the predicted game-level cell distribution; also batter-level |
| Foul count per batter-game | `description in foul/foul_tip/foul_bunt` | Poisson NLL, deviance |
| **Foul location** | **none**: Statcast has no hit coordinates for any foul ball (0 of 552k in 2021-24) | **Not scorable.** The v0 foul map is an unvalidated prior. |

Location is scored on a 90-cell grid: 18 spray-angle bins (5 deg) x 5 depth-beyond-wall bins
(0-10, 10-20, 20-30, 30-50, 50+ ft). Uniform guessing costs ln(90) = 4.50 nats per HR.
Real seat sections need `data/seatmaps/<venue>.json`, which does not exist yet, so section-level
accuracy cannot be measured.

## Backtest protocol (walk-forward, leak-free)

* **Data**: `ballpark research sync` fetches 2021-2026 Statcast (one parquet per day) into
  `data/research/statcast/` and game->venue(+temperature) / venue fence tables into `data/research/`.
  Separate from the production store so it can never affect the deploy job.
* **Lineups & starters are the *actual* ones**, derived from the pitch data (first nine batters per
  team-game in order of first PA; the pitcher of that team's first PA is the opposing starter).
  This is an upper bound on what production knows: it gets projected/posted lineups, which can be wrong.
  HRs by bench players are excluded from both prediction and truth.
* **Windows**: a game on day D uses only pitches from [D-365, D-1]. The league-recalibration windows
  (120/730 d) are also strictly prior-day.
* **Splits**: HR classifier + league priors fit on seasons <= `--train-through`.
  2024 = validation (all hyper-parameters tuned here, once). 2025 = holdout, run once with frozen
  parameters and a classifier trained through 2024. 2026 = a second out-of-sample season
  (classifier trained through 2025).
* **v0** (the production model, imported unmodified from `model.py`) is run on every 3rd day because
  it is ~10x slower. All models are compared on that same sample in the `*_sample` tables.
* Significance: paired bootstrap over game-days (rows within a day are correlated).

## Candidate model (v1)

Per batter-game, expected HR = `PA_exp x r_batter(park) x pitcher_effect x environment`:

1. **P(HR | ball, park)**: gradient-boosted classifier on exit velocity, launch angle, spray angle, wall
   distance at that angle, elevation, roof, venue id. Each of a batter's last-365-day batted balls is
   re-evaluated *in the game's park*: this is the counterfactual "would it have left here?".
   Hit distance is deliberately **not** a feature: with it the classifier is near-deterministic
   (AUC 0.9995) and its sum is just the batter's realised HR count, which predicts future HRs worse
   than a smoother contact-quality xHR (see ablation).
2. **Batter rate** = `(sum P(HR) + k x league park-xHR/PA) / (PA + k)`: shrunk toward the league's
   xHR rate *in that park*.
3. **Pitcher effect**: the starter's xHR/PA allowed vs the league (not vs the other starter), heavily
   shrunk, applied **only to the lineup he faces** and only to his expected share of PA
   (from his trailing batters-faced per start); the rest is league-average bullpen.
4. **PA expectation** from lineup slot x home/away (empirical, training years), not a hard-coded table.
5. **Platoon**: batter-vs-starter-hand multiplier relative to the batter's own trailing PA mix.
6. **Environment**: multiply by the league's trailing-120-day actual/expected HR ratio (juiced ball,
   weather drift; walk-forward).
7. **Location**: the batter's HR-weighted balls are spread over the 90 cells (Gaussian kernel in angle;
   depth beyond the wall), mixed with that park's historical HR-location prior, and normalised so the
   cells sum exactly to the batter-game HR expectation (v0's did not).
8. **Fouls**: shrunk batter foul/PA x shrunk pitcher foul/PA x PA expectation (counts only).

## Results (Poisson NLL per batter-game, lower is better; full seasons, ~43k lineup slots)

| Model | 2024 (validation) | 2025 (holdout) | 2026 |
|---|---|---|---|
| league rate | 0.3809 | 0.3912 | 0.3831 |
| batter only | 0.3755 | 0.3840 | 0.3783 |
| batter x park factor x pitcher | 0.3754 | 0.3833 | 0.3779 |
| **v1** | **0.3739** | **0.3828** | **0.3769** |

v1 minus batter-only (paired bootstrap, 95% CI): -0.0016 (-0.0023, -0.0010) / -0.0012 (-0.0018, -0.0006) /
-0.0013 (-0.0020, -0.0007). On the every-3rd-day sample, v0 is 0.040-0.062 NLL *worse* than v1 every year
(it over-predicts HRs by 16-20% and under-predicts fouls by ~38%). HR location (nats/HR): v1 beats v0
by ~0.13-0.16 at game level, but only ties a static per-park prior (2025: 4.226 vs 4.211).
Full tables: `output/backtests/<run>/report.md`.

## Running it

```bash
uv sync --group research
uv run --group research ballpark research sync
uv run --group research ballpark research backtest --name val2024 --test-start 2024-04-01 \
    --test-end 2024-09-30 --train-through 2023 --every 3 --tune
uv run --group research ballpark research backtest --name holdout2025 --test-start 2025-03-27 \
    --test-end 2025-09-28 --train-through 2024 --every 3 --params output/backtests/val2024/params.json
uv run --group research ballpark research report holdout2025
uv run --group research ballpark research ablate --test-year 2024
```
`uv sync` without `--group research` removes scikit-learn/scipy again (that is what CI does).

## CI isolation

* `research` is a non-default dependency group in `pyproject.toml`; the deploy workflow's plain `uv sync`
  does not install it.
* `cli.py` imports `ballpark.research` only inside `cmd_research`.
* `scripts/check_ci_isolation.py` (stdlib only, run by the workflow before export) fails the build if any
  non-research module imports `ballpark.research`, `sklearn` or `scipy`.

## Known limitations

* Fence distances are today's (`ballparks.json`); parks that moved walls (e.g. Camden Yards 2022,
  Citi Field 2025) are slightly wrong in history. Wall *height* is only learned implicitly via venue id.
* No weather/wind. Game temperature is in `data/research/games.parquet`, but a forecast would be needed
  at prediction time; the classifier ablation quantifies the upside.
* Foul-ball *location* cannot be validated (no coordinates) and foul-territory/netting geometry is not
  modelled, so "fouls into seats" remains an assumption.
* Production still serves v0. Promoting v1 needs a fitted-model artifact and an inference path
  (sklearn at inference, or an exported lookup table) that is reviewed separately.
