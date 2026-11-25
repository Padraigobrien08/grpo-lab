type Flat = Record<string, string>;

export function flatten(value: unknown, prefix = "", out: Flat = {}): Flat {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix] = JSON.stringify(value);
  }
  return out;
}

/** Config keys whose value is not the same in every run: what actually distinguishes runs. */
export function differingKeys(configs: Record<string, unknown>[], ignore: string[] = []): string[] {
  const flats = configs.map((c) => flatten(c));
  const keys = new Set(flats.flatMap((f) => Object.keys(f)));
  return [...keys]
    .filter((k) => !ignore.includes(k) && new Set(flats.map((f) => f[k])).size > 1)
    .sort();
}
