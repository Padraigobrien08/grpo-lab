import katex from "katex";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "math"; tex: string; display: boolean; source: string };

// Only unambiguous delimiters. Single `$` is never math here: GSM8K text is full of prices
// ("a $50 bill"), and treating those as math would garble the prose.
const OPENERS: { open: string; close: string; display: boolean }[] = [
  { open: "\\[", close: "\\]", display: true },
  { open: "$$", close: "$$", display: true },
  { open: "\\(", close: "\\)", display: false },
];

/** End index (exclusive) of a brace group starting at `start` ("{"), or -1 if unbalanced. */
function braceEnd(text: string, start: number): number {
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}" && --depth === 0) return i + 1;
  }
  return -1;
}

/**
 * Split a completion into prose and math. Unclosed delimiters (a sample truncated mid-formula)
 * stay as plain text rather than swallowing the rest of the answer. A bare \boxed{...} in prose
 * becomes inline math so the final answer renders like the rest.
 */
export function splitMath(text: string): Segment[] {
  const out: Segment[] = [];
  let buf = "";
  let i = 0;
  outer: while (i < text.length) {
    for (const d of OPENERS) {
      if (text.startsWith(d.open, i)) {
        const end = text.indexOf(d.close, i + d.open.length);
        if (end > 0) {
          if (buf) out.push({ kind: "text", text: buf });
          buf = "";
          const tex = text.slice(i + d.open.length, end);
          const source = text.slice(i, end + d.close.length);
          out.push({ kind: "math", tex, display: d.display, source });
          i = end + d.close.length;
          continue outer;
        }
      }
    }
    if (text.startsWith("\\boxed{", i)) {
      const end = braceEnd(text, i + "\\boxed".length);
      if (end > 0) {
        if (buf) out.push({ kind: "text", text: buf });
        buf = "";
        const source = text.slice(i, end);
        out.push({ kind: "math", tex: source, display: false, source });
        i = end;
        continue;
      }
    }
    buf += text[i++];
  }
  if (buf) out.push({ kind: "text", text: buf });
  return out;
}

/** KaTeX HTML, or null if the TeX doesn't parse (the caller shows the source instead). */
export function renderTex(tex: string, display: boolean): string | null {
  try {
    // trust: false (the default) keeps \href, \htmlClass and friends disabled: completions are
    // model output, not trusted markup.
    return katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "html" });
  } catch {
    return null;
  }
}
