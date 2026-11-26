type Flat = Record<string, string>;

export function flatten(value: unknown, prefix = "", out: Flat = {}): Flat {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix] = JSON.stringify(value);
  }
  return out;
}

/**
 * For each config, the [key, value] pairs where it departs from the most common value across
 * all configs: what makes that run different. Keys in `ignore` are skipped.
 */
export function configOutliers(
  configs: Record<string, unknown>[],
  ignore: string[] = [],
): [string, string][][] {
  const flats = configs.map((c) => flatten(c));
  const keys = [...new Set(flats.flatMap((f) => Object.keys(f)))].sort();
  const mode: Flat = {};
  for (const k of keys) {
    const counts = new Map<string, number>();
    for (const f of flats) counts.set(f[k], (counts.get(f[k]) ?? 0) + 1);
    mode[k] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  }
  return flats.map((f) =>
    keys.filter((k) => !ignore.includes(k) && f[k] !== mode[k]).map((k) => [k, f[k] ?? "unset"]),
  );
}
