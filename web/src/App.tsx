import { useEffect, useMemo, useState } from "react";
import { fetchDay, fetchGame, fetchRoot } from "./api";
import Field from "./Field";
import type { DayIndex, GameDetail, GameSummary, Kind, Side, Stats, TeamSide } from "./types";

const todayStr = () => new Date().toLocaleDateString("en-CA");

function useHash(): [string[], (p: string) => void] {
  const parse = () => location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [parts, setParts] = useState(parse);
  useEffect(() => {
    const f = () => setParts(parse());
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return [parts, (p) => { location.hash = p; }];
}

const fmtTime = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

const COLS: [string, string][] = [["PA", "PA"], ["EV", "EV"], ["Brl%", "Brl%"], ["HH%", "HH%"], ["HR", "HR"], ["Foul/PA", "F/PA"]];
const OPTIONAL = new Set(["PA", "HH%"]); // hidden on narrow screens so batter names fit
const fmt = (v: number | null | undefined) => (v == null ? "–" : String(v));

function Help({ open = false, days = 365 }: { open?: boolean; days?: number }) {
  return (
    <details className="help card" open={open}>
      <summary>How to use this page</summary>
      <ol>
        <li><b>Pick a day and a game.</b> Use the ‹ › arrows or the date menu at the top, then choose a game from the list.</li>
        <li><b>Read the heatmap.</b> It's the ballpark seen from above, home plate at the bottom. The yellow line is the outfield wall. Darker red means more expected home runs (or foul balls) land there.</li>
        <li><b>Switch what you see.</b> “Home runs” and “Foul balls” change the map. “Both / away / home bats” shows one team's hitters or both.</li>
        <li><b>Zoom in on a zone.</b> Click any row under “Hottest zones” to show only that zone on the map and its expected count. Click it again, or “show all”, to clear it.</li>
        <li><b>Check the lineups.</b> Each team's starter and batting order are listed with their Statcast numbers: PA (plate appearances), EV (average exit velocity, mph), Brl% (barrel rate), HH% (share of balls hit 95+ mph), HR, F/PA (foul balls per PA), and xHR (expected home runs in this game).</li>
      </ol>
      <h4>Good to know</h4>
      <ul>
        <li>Numbers are expected counts for this game, built from each batter's last {days} days of Statcast and the opposing starter's tendencies. They are estimates, not predictions of a specific seat.</li>
        <li>A lineup tagged <i>projected</i> hasn't been posted yet; it's the team's most recent lineup against a starter who throws with the same hand. It updates when the real one is posted.</li>
        <li>Foul-ball landing spots are estimated from exit velocity and launch angle, because Statcast doesn't record where most fouls land. Home runs ignore wall height and weather.</li>
        <li>Zones are angle-and-distance areas of the field, not stadium seat sections.</li>
        <li>Data refreshes several times a day as lineups come out.</li>
      </ul>
    </details>
  );
}

function Lineup({ side, fx }: { side: TeamSide; fx: Record<string, number> }) {
  const proj = side.lineup_source !== "posted";
  return (
    <section className="card">
      <h3>{side.team}</h3>
      {side.pitcher && (
        <p className="sp">
          <b>SP</b> {side.pitcher.name}
          <span className="muted"> · EV allowed {fmt(side.pitcher.stats["EV"])} · Brl% {fmt(side.pitcher.stats["Brl%"])} · HR {fmt(side.pitcher.stats["HR"])}</span>
        </p>
      )}
      {side.lineup.length === 0 ? <p className="muted">No lineup available.</p> : (
        <>
          {proj && <p className="badge">Lineup {side.lineup_source}</p>}
          <table>
            <thead><tr><th>#</th><th>Batter</th>{COLS.map(([k, l]) => <th key={l} className={OPTIONAL.has(k) ? "opt" : undefined}>{l}</th>)}<th>xHR</th></tr></thead>
            <tbody>
              {side.lineup.map((b) => (
                <tr key={b.id}>
                  <td>{b.slot}</td><td>{b.name}</td>
                  {COLS.map(([k]) => <td key={k} className={OPTIONAL.has(k) ? "opt" : undefined}>{fmt((b.stats as Stats)[k])}</td>)}
                  <td>{fx[b.name] != null ? fx[b.name].toFixed(2) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

function GameView({ g }: { g: GameDetail }) {
  const [kind, setKind] = useState<Kind>("HR");
  const [team, setTeam] = useState<"both" | Side>("both");
  const [zone, setZone] = useState<number | null>(null); // selected zone index
  const p = g.projection;
  const allCells = useMemo(() => {
    if (!p) return [];
    return team === "both" ? [...p.cells[kind].away, ...p.cells[kind].home] : p.cells[kind][team];
  }, [p, kind, team]);
  const cells = useMemo(() => (zone == null ? allCells : allCells.filter((c) => c[3] === zone)), [allCells, zone]);
  const scaleMax = useMemo(() => Math.max(...allCells.map((c) => c[2]), 1e-9), [allCells]);
  const total = p ? (team === "both" ? p.totals[kind].away + p.totals[kind].home : p.totals[kind][team]) : 0;
  const zoneTotal = zone == null ? total : cells.reduce((a, c) => a + c[2], 0);

  return (
    <>
      <header className="game-head">
        <h2>{g.away.team} @ {g.home.team}</h2>
        <p className="muted">{g.venue.name} · {fmtTime(g.time)} · {g.status} · capacity {g.venue.capacity?.toLocaleString()} · elev {g.venue.elevation_ft} ft</p>
      </header>
      <Help days={g.lookback_days} />
      {!p ? <p className="notice">{g.projection_note ?? "No projection available."}</p> : (
        <section className="card viz">
          <div className="controls">
            <div className="seg" role="tablist">
              {(["HR", "FOUL"] as Kind[]).map((k) => (
                <button key={k} className={kind === k ? "on" : ""} onClick={() => { setKind(k); setZone(null); }}>
                  {k === "HR" ? "Home runs" : "Foul balls (est.)"}
                </button>
              ))}
            </div>
            <div className="seg">
              {(["both", "away", "home"] as const).map((t) => (
                <button key={t} className={team === t ? "on" : ""} onClick={() => setTeam(t)}>
                  {t === "both" ? "Both" : g[t].team.split(" ").slice(-1)[0] + " bats"}
                </button>
              ))}
            </div>
          </div>
          <div className="viz-body">
            <Field kind={kind} cells={cells} fence={g.venue.fence} scaleMax={scaleMax} />
            <div className="side-panel">
              <div className="big">{zoneTotal.toFixed(kind === "HR" ? 2 : 1)}</div>
              <div className="muted">
                expected {kind === "HR" ? "home runs" : "foul balls"}
                {zone != null && <> in <b>{p.zones[kind][zone]?.zone}</b></>}
              </div>
              <h4>Hottest zones {zone != null && <button className="link" onClick={() => setZone(null)}>show all</button>}</h4>
              <ol className="zones">
                {p.zones[kind].slice(0, 12).map((z, i) => (
                  <li key={z.zone}>
                    <button className={`zone-btn ${zone === i ? "on" : ""}`} aria-pressed={zone === i}
                      onClick={() => setZone(zone === i ? null : i)}>
                      <span>{z.zone}</span><b>{z.exp.toFixed(2)}</b>
                    </button>
                  </li>
                ))}
              </ol>
              {kind === "FOUL" && <p className="muted small">Foul landing spots are estimated from exit velocity and launch angle; Statcast doesn’t record them.</p>}
            </div>
          </div>
        </section>
      )}
      <div className="two">
        <Lineup side={g.away} fx={p?.batters.HR ?? {}} />
        <Lineup side={g.home} fx={p?.batters.HR ?? {}} />
      </div>
      <p className="muted small">Based on {g.lookback_days} days of Statcast. Updated {new Date(g.generated_at).toLocaleString()}.</p>
    </>
  );
}

export default function App() {
  const [parts, go] = useHash();
  const [dates, setDates] = useState<string[]>([]);
  const [day, setDay] = useState<DayIndex | null>(null);
  const [game, setGame] = useState<GameDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { fetchRoot().then((r) => setDates(r.dates)).catch((e) => setErr(String(e))); }, []);

  const date = parts[0] ?? (dates.includes(todayStr()) ? todayStr() : dates[dates.length - 1]);
  const pk = parts[1] ? Number(parts[1]) : undefined;

  useEffect(() => {
    if (!date) return;
    setDay(null); setErr(null);
    fetchDay(date).then(setDay).catch(() => setErr(`No data for ${date}.`));
  }, [date]);

  useEffect(() => {
    if (!date || !pk) { setGame(null); return; }
    setGame(null);
    fetchGame(date, pk).then(setGame).catch(() => setErr("Game not found."));
  }, [date, pk]);

  const idx = dates.indexOf(date ?? "");
  return (
    <div className="app">
      <nav className="top">
        <h1>Ballpark Projections</h1>
        <div className="datebar">
          <button disabled={idx <= 0} onClick={() => go(dates[idx - 1])} aria-label="Previous day">‹</button>
          <select value={date ?? ""} onChange={(e) => go(e.target.value)}>
            {dates.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <button disabled={idx < 0 || idx >= dates.length - 1} onClick={() => go(dates[idx + 1])} aria-label="Next day">›</button>
        </div>
      </nav>
      <main>
        <aside className="games">
          {err && !day && <p className="notice">{err}</p>}
          {day?.games.map((s: GameSummary) => (
            <a key={s.game_pk} href={`#/${date}/${s.game_pk}`} className={`gcard ${pk === s.game_pk ? "sel" : ""}`}>
              <div className="teams">{s.away.team} <span className="muted">@</span> {s.home.team}</div>
              <div className="muted small">{fmtTime(s.time)} · {s.venue}</div>
              <div className="muted small">{s.away.pitcher ?? "TBD"} vs {s.home.pitcher ?? "TBD"}</div>
              <div className="chips">
                {s.has_projection ? (<><span className="chip">{s.hr_total} xHR</span><span className="chip">{s.foul_total} xFoul</span></>) : <span className="chip off">no projection</span>}
                {(s.away.lineup_source !== "posted" || s.home.lineup_source !== "posted") && s.has_projection && <span className="chip off">projected lineup</span>}
              </div>
            </a>
          ))}
          {day && day.games.length === 0 && <p className="muted">No games exported for this date.</p>}
        </aside>
        <div className="detail">
          {!pk && (<><p className="muted pad">Select a game to see its projection.</p><Help open /></>)}
          {pk && !game && !err && <p className="muted pad">Loading…</p>}
          {game && <GameView key={game.game_pk} g={game} />}
        </div>
      </main>
    </div>
  );
}
