# ballpark-projections

```bash
uv sync
uv run ballpark parks                         # all 30 parks -> data/ballparks.{json,csv}
uv run ballpark slate --date 2026-10-14       # games, lineups, SPs + Statcast summaries
uv run ballpark project --date 2026-10-14     # HR / foul seat-zone projections -> output/
uv run streamlit run src/ballpark/app.py      # UI: pick a date + game, see heatmaps
```

Data: MLB Stats API (parks, schedule, lineups, probable pitchers) and Statcast via pybaseball (cached in `data/cache/`).
Lineups only exist once teams post them (usually a few hours before first pitch); `project` skips games without them.

## Model assumptions (v0)
- HR: each batter's fly balls (LA 20-50 deg) re-projected into the game's park; a HR if distance > interpolated wall + 8 ft. Wall height ignored; fence points assumed evenly spaced at 15 deg.
- Fouls: Statcast lacks hit coordinates for most fouls, so landing spot is *estimated* (projectile from EV/LA, pull-biased direction). Output counts all fouls, not only those reaching seats; no foul-territory geometry yet.
- Pitcher adjustment: HR/PA and foul/PA vs slate mean, shrunk toward 1 (150 PA). No platoon splits, weather, or wind.
- Zones are generic angle x depth bins. For real seat sections, add `data/seatmaps/<venue_id>.json`: `[{"section": "134", "poly": [[x_ft, y_ft], ...]}]` (feet from home plate, +y to CF, +x to RF).

## Next
Seat maps per park, foul-territory/netting geometry, platoon splits, weather/wind, plot overlays.
