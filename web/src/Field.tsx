import { useEffect, useRef } from "react";
import type { Cell, Kind } from "./types";

const ANGLES = [-45, -30, -15, 0, 15, 30, 45];
const CELL = 10;

function fenceAt(angle: number, fence: number[]): number {
  const a = Math.max(-45, Math.min(45, angle));
  const i = Math.min(5, Math.floor((a + 45) / 15));
  const t = (a - ANGLES[i]) / 15;
  return fence[i] + t * (fence[i + 1] - fence[i]);
}
const polar = (angle: number, d: number): [number, number] => {
  const r = (angle * Math.PI) / 180;
  return [d * Math.sin(r), d * Math.cos(r)];
};

// ramp from transparent yellow to deep red
function color(t: number): string {
  const h = 55 - 55 * t;
  return `hsla(${h}, 95%, ${60 - 20 * t}%, ${0.25 + 0.7 * t})`;
}

export default function Field({ kind, cells, fence }: { kind: Kind; cells: Cell[]; fence: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current!;
    const dpr = window.devicePixelRatio || 1;
    // world window in feet: foul view shows territory behind the plate
    const [x0, x1, y0, y1] = kind === "FOUL" ? [-420, 420, -130, 440] : [-300, 300, -20, 470];
    const W = c.clientWidth;
    const H = (W * (y1 - y0)) / (x1 - x0);
    c.width = W * dpr; c.height = H * dpr; c.style.height = `${H}px`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = W / (x1 - x0);
    const X = (x: number) => (x - x0) * s;
    const Y = (y: number) => H - (y - y0) * s;

    const styles = getComputedStyle(document.documentElement);
    ctx.fillStyle = styles.getPropertyValue("--field-bg") || "#2f6b39";
    ctx.fillRect(0, 0, W, H);

    // fair territory
    ctx.beginPath(); ctx.moveTo(X(0), Y(0));
    for (let a = -45; a <= 45; a += 1) { const [x, y] = polar(a, fenceAt(a, fence)); ctx.lineTo(X(x), Y(y)); }
    ctx.closePath(); ctx.fillStyle = styles.getPropertyValue("--field-fair") || "#3f8a4b"; ctx.fill();

    // infield dirt
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(63.6), Y(63.6));
    ctx.lineTo(X(0), Y(127.3)); ctx.lineTo(X(-63.6), Y(63.6)); ctx.closePath();
    ctx.fillStyle = "rgba(200,161,101,.65)"; ctx.fill();

    // heat
    const max = Math.max(...cells.map((c) => c[2]), 1e-9);
    ctx.filter = "blur(3px)";
    for (const [x, y, e] of cells) {
      ctx.fillStyle = color(Math.sqrt(e / max));
      ctx.fillRect(X(x - CELL / 2), Y(y + CELL / 2), CELL * s + 0.5, CELL * s + 0.5);
    }
    ctx.filter = "none";

    // lines + wall
    ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.2;
    for (const sg of [-1, 1]) {
      const [ex, ey] = polar(sg * 45, fenceAt(sg * 45, fence));
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(ex * 1.0), Y(ey)); ctx.stroke();
    }
    ctx.strokeStyle = "#ffd54a"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let a = -45; a <= 45; a += 1) {
      const [x, y] = polar(a, fenceAt(a, fence));
      a === -45 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.font = "11px system-ui";
    [-45, -22.5, 0, 22.5, 45].forEach((a) => {
      const d = fenceAt(a, fence), [x, y] = polar(a, d);
      ctx.fillText(`${Math.round(d)}`, X(x) - 9, Y(y) + (a === 0 ? 14 : 14));
    });
  }, [kind, cells, fence]);

  return <canvas ref={ref} className="field" role="img" aria-label={`${kind} landing heatmap`} />;
}
