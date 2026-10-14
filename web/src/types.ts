export type Stats = Record<string, number | null>;
export type Kind = "HR" | "FOUL";
export type Side = "away" | "home";
export type Cell = [number, number, number, number]; // x ft, y ft, expected count, zone index (-1 = other)

export interface RootIndex { dates: string[]; updated: string }
export interface GameSummary {
  game_pk: number; time: string; status: string; venue: string;
  away: { team: string; pitcher: string | null; lineup_source: string };
  home: { team: string; pitcher: string | null; lineup_source: string };
  has_projection: boolean; hr_total: number | null; foul_total: number | null; note: string | null;
}
export interface DayIndex { date: string; games: GameSummary[]; generated_at: string }

export interface Batter { slot: number; id: number; name: string; stats: Stats }
export interface TeamSide {
  team: string;
  pitcher: { id: number; name: string; stats: Stats } | null;
  lineup: Batter[];
  lineup_source: string;
}
export interface Projection {
  totals: Record<Kind, Record<Side, number>>;
  cells: Record<Kind, Record<Side, Cell[]>>;
  zones: Record<Kind, { zone: string; exp: number }[]>;
  batters: Record<Kind, Record<string, number>>;
}
export interface GameDetail {
  game_pk: number; date: string; time: string; status: string;
  venue: { id: number; name: string; capacity: number; elevation_ft: number; roof: string; fence: number[] };
  away: TeamSide; home: TeamSide; lookback_days: number; generated_at: string;
  projection: Projection | null; projection_note?: string;
}
