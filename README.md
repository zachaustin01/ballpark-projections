# ballpark-projections

```bash
uv sync
uv run ballpark parks                         # all 30 parks -> data/ballparks.{json,csv}
uv run ballpark slate --date 2026-10-14       # games, lineups, SPs + Statcast summaries
uv run ballpark project --date 2026-10-14     # HR / foul seat-zone projections -> output/
uv run ballpark export                        # write static JSON for the site -> web/public/data/
cd web && npm install && npm run dev          # UI at http://localhost:5173
```

Data: MLB Stats API (parks, schedule, lineups, probable pitchers) and Statcast via pybaseball. Statcast is kept as a local league-wide store (one parquet per game day in `data/statcast/`, ~11 MB/year): `uv run ballpark sync` backfills 365 days in ~30 s, and every command auto-syncs missing days.
Lineups only exist once teams post them (usually a few hours before first pitch); `project` skips games without them.

## Model assumptions (v0)
- HR: each batter's fly balls (LA 20-50 deg) re-projected into the game's park; a HR if distance > interpolated wall + 8 ft. Wall height ignored; fence points assumed evenly spaced at 15 deg.
- Fouls: Statcast lacks hit coordinates for most fouls, so landing spot is *estimated* (projectile from EV/LA, pull-biased direction). Output counts all fouls, not only those reaching seats; no foul-territory geometry yet.
- Pitcher adjustment: HR/PA and foul/PA vs slate mean, shrunk toward 1 (150 PA). No platoon splits, weather, or wind.
- Zones are generic angle x depth bins. For real seat sections, add `data/seatmaps/<venue_id>.json`: `[{"section": "134", "poly": [[x_ft, y_ft], ...]}]` (feet from home plate, +y to CF, +x to RF).

## Research / backtesting (local only)
Model development lives in `src/ballpark/research/` and is **never run in CI**: it needs the optional `research` dependency group, the deploy workflow runs a plain `uv sync` (which skips it), and `scripts/check_ci_isolation.py` (run by the workflow) fails the build if CI-path code imports anything from `research`.

```bash
uv sync --group research
uv run --group research ballpark research sync                       # 2021-2026 pitch store -> data/research/ (gitignored, ~150 MB)
uv run --group research ballpark research backtest --name val2024 \
    --test-start 2024-04-01 --test-end 2024-09-30 --train-through 2023 --tune     # tune on validation year
uv run --group research ballpark research backtest --name holdout2025 \
    --test-start 2025-03-27 --test-end 2025-09-28 --train-through 2024 \
    --params output/backtests/val2024/params.json                                 # frozen params, run once
```
Results land in `output/backtests/<name>/` (metric CSVs, `predictions.parquet`, `report.md`). Details: `docs/MODELING.md`.

## Next
Seat maps per park, foul-territory/netting geometry, platoon splits, weather/wind, plot overlays.
