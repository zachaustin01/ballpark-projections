import type { DayIndex, GameDetail, RootIndex } from "./types";

const base = import.meta.env.BASE_URL + "data/";

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

export const fetchRoot = () => getJson<RootIndex>("index.json");
export const fetchDay = (d: string) => getJson<DayIndex>(`${d}/index.json`);
export const fetchGame = (d: string, pk: number) => getJson<GameDetail>(`${d}/${pk}.json`);
